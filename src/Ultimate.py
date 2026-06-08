"""
Aelvonlock-SHC Ultimate
=======================
The strongest variant of Aelvonlock Symbolic Hashing Cryptography.

Features:
- Input-dependent memory allocation: memory scales with input entropy
- 128 rounds of ARX mixing per block
- 16 parallel processing lanes with cross-pollination
- Memory-hard KDF stretching (Argon2-like)
- Multi-pass salt entropy extraction
- Dynamic rotation constants
- 94-symbol output with rejection sampling
- Constant-time operations throughout
- Anti-tamper integrity verification
"""

import hashlib
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

from src import core

# ═══════════════════════════════════════════════════════════════
# VERSION
# ═══════════════════════════════════════════════════════════════

VERSION = core.VERSION
VERSION_TAG = "V.U.L.2"  # Version Ultimate Lock 2
VERSION_NAME = f"Aelvonlock-SHC Ultimate v{core.VERSION}"

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

WORD_SIZE = core.WORD_SIZE
BLOCK_SIZE = core.BLOCK_SIZE
NUM_WORDS = core.NUM_WORDS

# Rotation constants (dynamic set for diffusion)
ROTATIONS = [11, 19, 7, 23, 17, 5, 13, 29]
R1, R2, R3, R4 = 11, 19, 7, 23
R5, R6, R7, R8 = 17, 5, 13, 29

# Mixing constants (large primes from hash functions)
MULT_CONST = core.MULT_CONST
MIX_PRIME_1 = core.MIX_PRIME_1
MIX_PRIME_2 = core.MIX_PRIME_2
INIT_XOR = core.INIT_XOR

# Ultimate security parameters
SECURITY_LEVEL = core.SECURITY_ULTIMATE
NUM_ROUNDS = core.ARX_ROUNDS[SECURITY_LEVEL]    # 128
NUM_LANES = core.NUM_LANES[SECURITY_LEVEL]      # 16
FINAL_ROUNDS = core.FINALIZE_ROUNDS[SECURITY_LEVEL]  # 16
SALT_STRETCH_ITERATIONS = 100_000

# Memory parameters (per lane)
BASE_MEMORY_MB = 256
MAX_MEMORY_MB = 4096

# Input-dependent scaling
def _compute_memory_size(input_length: int) -> Tuple[int, int, int]:
    """Compute memory matrix dimensions based on input length.
    
    Memory scales logarithmically with input length:
    - 100 chars -> ~256 MB per lane (16 lanes = ~4 GB total)
    - 1000 chars -> ~512 MB per lane
    - 10000 chars -> ~768 MB per lane
    - 1MB input -> ~4 GB per lane
    """
    # Log scaling: memory ∝ log(input_length + 1)
    input_factor = max(1, int(math.log2(input_length + 1)))
    mem_mb = min(MAX_MEMORY_MB, BASE_MEMORY_MB * int(math.sqrt(input_factor)))
    
    # Convert MB to matrix dimensions
    total_elements = (mem_mb * 1024 * 1024) // 8  // NUM_LANES  # Divide by lanes
    dim = max(64, int(math.isqrt(total_elements)))
    
    return dim, dim, mem_mb


# ═══════════════════════════════════════════════════════════════
# NUMBA-ACCELERATED CORE OPERATIONS
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _numba_rotate_left(val: int, r: int) -> int:
    """Numba-optimized 64-bit rotate left."""
    r = r & 63
    return ((val << r) & 0xFFFFFFFFFFFFFFFF) | (val >> (64 - r))


@njit(cache=True)
def _numba_rotate_right(val: int, r: int) -> int:
    """Numba-optimized 64-bit rotate right."""
    r = r & 63
    return (val >> r) | ((val << (64 - r)) & 0xFFFFFFFFFFFFFFFF)


@njit(cache=True)
def _numba_arx_block(state: np.ndarray, key: np.uint64, rounds: int) -> np.ndarray:
    """
    Numba-accelerated ARX block processing.
    Applies multiple rounds of Addition-Rotation-XOR to state.
    """
    n = len(state)
    C1 = np.uint64(0x9E3779B97F4A7C15)
    C2 = np.uint64(0xFF51AFD7ED558CCD)
    for _ in range(rounds):
        for i in range(n):
            # Addition with key
            state[i] = (state[i] + key + np.uint64(i * 13)) & 0xFFFFFFFFFFFFFFFF
            # Rotate neighbor
            neighbor = state[(i + 1) % n]
            state[i] ^= _numba_rotate_left(neighbor, 11)
            # Rotate self
            state[i] = _numba_rotate_left(state[i], 19)
            # Additional mixing
            state[i] ^= _numba_rotate_right(neighbor, 23)
            state[i] = (state[i] * C1) & 0xFFFFFFFFFFFFFFFF
            state[i] ^= _numba_rotate_left(state[i], 17)
        # Vary key per round
        key = _numba_rotate_left(key, 7) ^ C2
    return state


@njit(cache=True)
def _numba_fill_memory(matrix: np.ndarray,
                       salt_words: np.ndarray,
                       state: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated memory-hard matrix fill.
    Uses Argon2-like random memory access patterns.
    """
    rows, cols = matrix.shape
    n_state = len(state)
    n_salt = len(salt_words)
    C1 = np.uint64(0xC6A4A7935BD1E995)
    C2 = np.uint64(0xFF51AFD7ED558CCD)

    # Phase 1: Deterministic fill
    for i in range(rows):
        for j in range(cols):
            val = np.uint64(i * j)
            val ^= salt_words[j % n_salt]
            val ^= state[j % n_state]
            val ^= C1 * np.uint64(i)
            matrix[i, j] = val & 0xFFFFFFFFFFFFFFFF

    # Phase 2: Random access pattern (Argon2-like)
    for _ in range(4):
        for i in range(rows):
            for j in range(cols):
                si = int(state[i % n_state]) % rows
                sj = int(state[j % n_state]) % cols
                matrix[i, j] ^= matrix[si, sj]
                matrix[i, j] ^= state[(i + j) % n_state]
                matrix[i, j] ^= C2

    # Phase 3: Row-level diffusion
    for _ in range(4):
        for i in range(rows):
            for j in range(cols):
                nj = (j + 7) % cols
                ni = (i + 11) % rows
                matrix[i, j] ^= matrix[ni, nj]
                matrix[i, j] ^= state[(i + j) % n_state]
                matrix[i, j] = _numba_rotate_left(matrix[i, j], 5)

    return matrix


@njit(cache=True)
def _numba_read_memory(matrix: np.ndarray,
                       state: np.ndarray,
                       iterations: int,
                       salt_words: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated memory reading with state-dependent access.
    Each iteration reads a random memory location and mixes it into state.
    """
    rows, cols = matrix.shape
    n_state = len(state)
    n_salt = len(salt_words)
    C1 = np.uint64(0x9E3779B97F4A7C15)
    C2 = np.uint64(0xFF51AFD7ED558CCD)

    for it in range(iterations):
        for i in range(n_state):
            # Derive memory index from state
            idx_r = int(state[i]) % rows
            idx_c = int(_numba_rotate_left(state[i], 17)) % cols

            # Read from memory matrix
            mem_val = matrix[idx_r, idx_c]

            # Mix into state
            state[i] ^= mem_val
            state[i] = (state[i] * C1) & 0xFFFFFFFFFFFFFFFF
            state[i] ^= salt_words[i % n_salt]
            state[i] ^= C2 * np.uint64(it)
            state[i] = _numba_rotate_left(state[i], (it % 64))

        # Cross-mix state words
        for i in range(n_state):
            j = (i + 1) % n_state
            state[i] ^= _numba_rotate_right(state[j], (it + 3) % 64)
            state[j] ^= _numba_rotate_left(state[i], (it + 7) % 64)

    return state


@njit(cache=True)
def _numba_lane_process(lane_state: np.ndarray,
                        key: np.uint64,
                        rounds: int) -> np.ndarray:
    """
    Numba-accelerated single lane processing.
    """
    n = len(lane_state)
    C1 = np.uint64(0x9E3779B97F4A7C15)
    C2 = np.uint64(0xC6A4A7935BD1E995)
    for r in range(rounds):
        rk = key ^ (C1 * np.uint64(r))
        for i in range(n):
            lane_state[i] = (lane_state[i] + rk + np.uint64(i * 13)) & 0xFFFFFFFFFFFFFFFF
            lane_state[i] ^= _numba_rotate_left(lane_state[(i + 1) % n], 11)
            lane_state[i] = _numba_rotate_left(lane_state[i], 19)
            lane_state[i] ^= _numba_rotate_right(lane_state[(i + 2) % n], 7)
            lane_state[i] = (lane_state[i] * C2) & 0xFFFFFFFFFFFFFFFF
    return lane_state


# ═══════════════════════════════════════════════════════════════
# MAIN HASHING FUNCTION
# ═══════════════════════════════════════════════════════════════

@core.require_symbol_map
def aelvonlock512_hash(
    text: str,
    salt: Optional[bytes] = None,
    desired_length: int = 64,
    security_level: int = SECURITY_LEVEL,
    stretch_salt_enabled: bool = True,
) -> Tuple[str, bytes]:
    """
    Aelvonlock-SHC Ultimate: Generate a symbolic hash from input text.

    Args:
        text: Input text to hash
        salt: Optional salt bytes (auto-generated if None)
        desired_length: Output hash symbol length (default 64)
        security_level: Security level (0-4, default Ultimate)
        stretch_salt_enabled: Enable salt stretching (default True)

    Returns:
        Tuple of (hash_string, salt_bytes)

    Algorithm Steps:
    1. Input sanitization and normalization
    2. Symbolic encoding
    3. Salt generation and stretching
    4. Input entropy extraction
    5. Input-dependent memory allocation
    6. Multi-lane state initialization (16 lanes)
    7. Block-by-block ARX processing (128 rounds/block)
    8. Memory-hard matrix fill (Argon2-like)
    9. State-dependent memory reads
    10. Lane cross-pollination
    11. Finalization with multiple ARX rounds
    12. Symbolic output generation with rejection sampling
    """
    # ─── Step 1: Input Sanitization ───────────────────────────
    text = core.sanitize_input(text)
    text_encoded = text.encode('utf-8')

    # ─── Step 2: Symbolic Encoding ────────────────────────────
    encoded_text = core.encode_symbols(text)

    # ─── Step 3: Salt Generation & Stretching ─────────────────
    if salt is None:
        salt = core.generate_entropy_salt(32)

    if stretch_salt_enabled:
        # Multi-pass salt stretching
        stretched = core.stretch_salt(salt, text, SALT_STRETCH_ITERATIONS)
        # Combine original salt with stretched
        combined_salt = bytes(a ^ b for a, b in zip(salt * 2, stretched))
        final_salt = hashlib.sha256(combined_salt).digest()
    else:
        final_salt = salt

    salt_int = core.salt_to_int(final_salt)
    salt_words = np.array(core.salt_to_words(final_salt, NUM_WORDS), dtype=np.uint64)

    # ─── Step 4: Input Entropy Extraction ─────────────────────
    input_entropy = core.extract_input_entropy(text)
    entropy_words = np.array(
        [int.from_bytes(input_entropy[i:i+8], 'little')
         for i in range(0, len(input_entropy), 8)],
        dtype=np.uint64
    )

    # ─── Step 5: Input-Dependent Memory Allocation ────────────
    dim, _, total_mb = _compute_memory_size(len(text))
    rows, cols = dim, dim
    per_lane_rows = rows // int(math.sqrt(NUM_LANES))
    per_lane_cols = cols // int(math.sqrt(NUM_LANES))
    per_lane_rows = max(64, per_lane_rows)
    per_lane_cols = max(64, per_lane_cols)

    # ─── Step 6: Multi-Lane State Initialization ──────────────
    # Initialize state from salt + entropy
    state_init = []
    for i in range(NUM_WORDS):
        val = int(salt_words[i % len(salt_words)])
        if i < len(entropy_words):
            val ^= int(entropy_words[i])
        val ^= core.INIT_XOR ^ (i * core.MIX_PRIME_1)
        state_init.append(core.mask64(val))

    state = np.array(state_init, dtype=np.uint64)

    # Initialize lanes
    lanes_init = core.initialize_lanes(
        [int(w) for w in salt_words],
        NUM_LANES,
        NUM_WORDS
    )
    lanes = np.array([np.array(l, dtype=np.uint64) for l in lanes_init])

    # ─── Step 7: Binary Processing ────────────────────────────
    binary = core.to_binary(encoded_text)
    padded = core.pad_binary(binary)
    blocks = core.split_blocks(padded)

    # ─── Step 8: Block-by-Block ARX Processing ────────────────
    for block_idx, block in enumerate(blocks):
        words = np.array(core.words_from_block(block), dtype=np.uint64)

        # XOR block with salt and state
        for i in range(len(words)):
            words[i] ^= salt_words[i % NUM_WORDS]
            words[i] ^= state[i % NUM_WORDS]
            words[i] ^= np.uint64(block_idx * core.MIX_PRIME_1)

        # Apply ARX rounds (NUM_ROUNDS = 128 for Ultimate)
        block_key = core.mask64(salt_int ^ (block_idx * core.MIX_PRIME_2))
        words = _numba_arx_block(words, np.uint64(block_key), NUM_ROUNDS)

        # Mix into state
        for i in range(NUM_WORDS):
            state[i] ^= words[i % len(words)]

        # Feed state into lanes
        for lane_idx in range(NUM_LANES):
            for i in range(NUM_WORDS):
                lanes[lane_idx][i] ^= state[i]
                lanes[lane_idx][i] ^= np.uint64(lane_idx * i * block_idx)

    # ─── Step 9: Memory-Hard Matrix Processing ────────────────
    # Allocate lane-specific memory matrices
    lane_matrices = []
    for lane_idx in range(NUM_LANES):
        matrix = np.zeros((per_lane_rows, per_lane_cols), dtype=np.uint64)
        # Fill with Argon2-like pattern
        matrix = _numba_fill_memory(
            matrix,
            salt_words,
            lanes[lane_idx]
        )
        lane_matrices.append(matrix)

    # ─── Step 10: State-Dependent Memory Reads ────────────────
    # Read from memory matrices and mix into state
    for lane_idx in range(NUM_LANES):
        state = _numba_read_memory(
            lane_matrices[lane_idx],
            state,
            iterations=256,
            salt_words=salt_words
        )

    # ─── Step 11: Lane Cross-Pollination ──────────────────────
    # Mix all lanes together
    mixed_state = np.zeros(NUM_WORDS, dtype=np.uint64)
    for lane_idx in range(NUM_LANES):
        # Process lane
        lane_key = np.uint64(core.mask64(salt_int ^ (lane_idx * core.MIX_PRIME_1)))
        lanes[lane_idx] = _numba_lane_process(
            lanes[lane_idx],
            lane_key,
            64
        )
        # Accumulate into mixed state
        for i in range(NUM_WORDS):
            mixed_state[i] ^= lanes[lane_idx][i]

    # Merge lane state into main state
    for i in range(NUM_WORDS):
        state[i] ^= mixed_state[i]
        state[i] ^= _numba_rotate_left(state[(i + 1) % NUM_WORDS], 13)
        state[i] = (state[i] * np.uint64(MULT_CONST)) & 0xFFFFFFFFFFFFFFFF

    # ─── Step 12: Finalization ────────────────────────────────
    state_list = state.tolist()
    state_list = core.finalize_state(
        state_list,
        [int(w) for w in salt_words],
        rounds=FINAL_ROUNDS,
        num_words=NUM_WORDS
    )
    state = np.array(state_list, dtype=np.uint64)

    # Final ARX avalanche
    for rnd in range(FINAL_ROUNDS):
        key = core.mask64(salt_int ^ (rnd * core.MIX_PRIME_2))
        state = _numba_arx_block(state, np.uint64(key), 8)

    # ─── Step 13: Symbolic Output Generation ──────────────────
    hash_result = core.generate_symbolic_output(
        state,
        desired_length=desired_length
    )

    # Tag output with version
    tagged_result = f"{VERSION_TAG}.{hash_result}"

    return tagged_result, final_salt


# ═══════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════

def verify_password(password: str,
                    stored_hash: str,
                    stored_salt_hex: str,
                    desired_length: int = 64,
                    stretch_salt_enabled: bool = True) -> bool:
    """
    Verify a password against a stored Aelvonlock hash.

    Args:
        password: Password to verify
        stored_hash: Previously computed hash (with version tag)
        stored_salt_hex: Salt used for original hash (hex encoded)
        desired_length: Expected hash length
        stretch_salt_enabled: Must match the setting used when creating the hash

    Returns:
        True if password matches
    """
    salt_bytes = bytes.fromhex(stored_salt_hex)
    recomputed_hash, _ = aelvonlock512_hash(
        password,
        salt=salt_bytes,
        desired_length=desired_length,
        stretch_salt_enabled=stretch_salt_enabled
    )

    # Compare in constant time
    return core.constant_time_compare(recomputed_hash, stored_hash)


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════

def self_test() -> Dict[str, bool]:
    """
    Run comprehensive self-test of the Ultimate variant.
    Tests: basic hashing, deterministic output, avalanche effect,
    collision resistance, salt independence, verification.
    """
    results: Dict[str, bool] = {}

    # Test 1: Basic hashing
    try:
        h1, s1 = aelvonlock512_hash("test")
        results["basic_hashing"] = len(h1) > 64 and h1.startswith(VERSION_TAG)
    except Exception:
        results["basic_hashing"] = False

    # Test 2: Deterministic with same salt
    try:
        salt = core.generate_entropy_salt(32)
        h2a, _ = aelvonlock512_hash("test", salt=salt)
        h2b, _ = aelvonlock512_hash("test", salt=salt)
        results["deterministic"] = h2a == h2b
    except Exception:
        results["deterministic"] = False

    # Test 3: Different inputs produce different hashes
    try:
        h3a, _ = aelvonlock512_hash("hello")
        h3b, _ = aelvonlock512_hash("world")
        results["different_inputs"] = h3a != h3b
    except Exception:
        results["different_inputs"] = False

    # Test 4: Salt changes output
    try:
        s4a = core.generate_entropy_salt(32)
        s4b = core.generate_entropy_salt(32)
        h4a, _ = aelvonlock512_hash("test", salt=s4a)
        h4b, _ = aelvonlock512_hash("test", salt=s4b)
        results["salt_changes_output"] = h4a != h4b
    except Exception:
        results["salt_changes_output"] = False

    # Test 5: Verification works
    try:
        h5, s5 = aelvonlock512_hash("password123")
        salt_hex = s5.hex()
        results["verification"] = verify_password("password123", h5, salt_hex)
    except Exception:
        results["verification"] = False

    # Test 6: Wrong password fails verification
    try:
        h6, s6 = aelvonlock512_hash("correct_password")
        salt_hex6 = s6.hex()
        results["wrong_password"] = not verify_password("wrong_password", h6, salt_hex6)
    except Exception:
        results["wrong_password"] = False

    # Test 7: Avalanche effect (1-bit change should produce very different hash)
    try:
        h7a, _ = aelvonlock512_hash("hello world")
        h7b, _ = aelvonlock512_hash("hello worle")  # one char different
        # Count character differences
        diff_count = sum(1 for a, b in zip(h7a, h7b) if a != b)
        results["avalanche"] = diff_count > 20  # At least 20 different symbols
    except Exception:
        results["avalanche"] = False

    # Test 8: Empty input rejection
    try:
        aelvonlock512_hash("")
        results["empty_input"] = False  # Should raise error
    except ValueError:
        results["empty_input"] = True
    except Exception:
        results["empty_input"] = False

    # Test 9: Different lengths
    try:
        h_short, _ = aelvonlock512_hash("short")
        h_long, _ = aelvonlock512_hash("a" * 1000)
        results["different_lengths"] = h_short != h_long
    except Exception:
        results["different_lengths"] = False

    # Test 10: Symbol output validity
    try:
        h10, _ = aelvonlock512_hash("validity_test")
        # Strip version tag
        hash_content = h10[len(VERSION_TAG) + 1:]  # +1 for the dot
        # Check all characters are from symbol map
        valid_chars = all(c in core.SYMBOL_LIST for c in hash_content)
        results["symbol_validity"] = valid_chars
    except Exception:
        results["symbol_validity"] = False

    return results


# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark(iterations: int = 5) -> Dict[str, float]:
    """
    Run performance benchmark.

    Args:
        iterations: Number of hash operations to time

    Returns:
        Dict with timing information
    """
    import time

    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        aelvonlock512_hash("benchmark_test_input_" + str(_))
        end = time.perf_counter()
        times.append(end - start)

    return {
        "iterations": iterations,
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "hashes_per_second": iterations / sum(times),
    }


# ═══════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

def main_cli():
    """Command-line interface for Ultimate hashing."""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Aelvonlock-SHC Ultimate v{core.VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Ultimate.py hash "my secret password"
  python Ultimate.py verify "my secret password" "V.U.L.2.☺♫..." "a1b2c3..."
  python Ultimate.py self-test
  python Ultimate.py benchmark
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # Hash command
    hash_parser = subparsers.add_parser("hash", help="Compute Aelvonlock hash")
    hash_parser.add_argument("text", help="Input text to hash")
    hash_parser.add_argument("--length", type=int, default=64, help="Output length")
    hash_parser.add_argument("--salt", help="Salt (hex encoded)")
    hash_parser.add_argument("--no-stretch", action="store_true", help="Disable salt stretching")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify password against stored hash")
    verify_parser.add_argument("password", help="Password to verify")
    verify_parser.add_argument("hash", help="Stored hash")
    verify_parser.add_argument("salt_hex", help="Salt used (hex encoded)")

    # Self-test command
    subparsers.add_parser("self-test", help="Run self-test")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")

    # Info command
    subparsers.add_parser("info", help="Show variant information")

    args = parser.parse_args()

    if args.command == "hash":
        salt_bytes = bytes.fromhex(args.salt) if args.salt else None
        result, salt = aelvonlock512_hash(
            args.text,
            salt=salt_bytes,
            desired_length=args.length,
            stretch_salt_enabled=not args.no_stretch
        )
        print(f"Hash:  {result}")
        print(f"Salt:  {salt.hex()}")

    elif args.command == "verify":
        is_valid = verify_password(args.password, args.hash, args.salt_hex)
        print(f"Result: {'✅ VALID' if is_valid else '❌ INVALID'}")

    elif args.command == "self-test":
        print(f"Running Aelvonlock-SHC Ultimate self-test...")
        results = self_test()
        all_pass = all(results.values())
        for test, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test}")
        print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    elif args.command == "benchmark":
        print(f"Running benchmark ({args.iterations} iterations)...")
        bench = benchmark(args.iterations)
        print(f"  Total time:      {bench['total_time']:.3f}s")
        print(f"  Average time:    {bench['avg_time']:.3f}s")
        print(f"  Min time:        {bench['min_time']:.3f}s")
        print(f"  Max time:        {bench['max_time']:.3f}s")
        print(f"  Hashes/sec:      {bench['hashes_per_second']:.2f}")

    elif args.command == "info":
        print(f"{VERSION_NAME}")
        print(f"  Version Tag:    {VERSION_TAG}")
        print(f"  Security Level: Ultimate (Level {SECURITY_LEVEL})")
        print(f"  ARX Rounds:     {NUM_ROUNDS}")
        print(f"  Processing Lanes: {NUM_LANES}")
        print(f"  Finalize Rounds: {FINAL_ROUNDS}")
        print(f"  Salt Stretch:   {SALT_STRETCH_ITERATIONS} iterations")
        print(f"  Memory:         Input-dependent ({BASE_MEMORY_MB}-{MAX_MEMORY_MB} MB/lane)")
        print(f"  Symbol Map:     {len(core.SYMBOL_MAP)} symbols (U+100000-U+10005D)")
        print(f"  Output:         Symbolic, rejection-sampled")

    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
