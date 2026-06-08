"""
Aelvonlock-SHC Maxlock
=======================
Maximum security variant for enterprise systems handling highly sensitive data.
Uses ~1 GB memory and 32+ ARX rounds with full multi-lane processing (4+ lanes).

Security: Maximum (highest resistance against all attack vectors)
Focus: Maximum cryptographic strength at the cost of memory/CPU
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from src import core

# ---------------- Constants ---------------- 
VERSION_TAG = "V.M.L.1"
NUM_WORDS = core.NUM_WORDS  # 8
ROUND_COUNT = 32
FINALIZE_ROUNDS = 5
NUM_ROUNDS = 128

MEM_MATRIX_ROWS = 8192
MEM_MATRIX_COLS = 16384

# ---------------- Numba-Accelerated Operations ----------------

@njit(cache=True)
def _process_matrix(matrix, salt_int, state):
    rows, cols = matrix.shape
    n_state = len(state)

    # Fill
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = np.uint64((i * j ^ salt_int) & 0xFFFFFFFFFFFFFFFF)

    # Heavy passes
    for _ in range(16):
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] ^= matrix[i, j] ^ state[i % n_state] ^ np.uint64(core.MIX_PRIME_1)

    # ARX-style diffusion
    for rnd in range(4):
        for i in range(rows):
            for j in range(cols):
                idx2 = (j + rnd) % cols
                matrix[i, j] ^= matrix[i, idx2] ^ state[(i + j + rnd) % n_state]

    return matrix


# ---------------- Main Hashing ----------------

def aelvonlock512_hash(text: str, salt: Optional[bytes] = None,
                       desired_length: int = 64) -> Tuple[str, bytes]:
    """Aelvonlock Maxlock hash function - maximum security."""
    text = core.sanitize_input(text)
    encoded = core.encode_symbols(text)
    binary = core.to_binary(encoded)
    padded = core.pad_binary(binary)

    # Salt mutation
    def mutate_salt(salt_bytes: bytes) -> bytes:
        pad = len(salt_bytes)
        entropy = bytearray(pad)
        for i in range(pad):
            entropy[i] = ((salt_bytes[i] * (i + 7)) ^
                          ((salt_bytes[(i - 1) % pad] + i * 19) & 0xFF)) % 256
        return bytes(entropy)

    if salt is None:
        salt = core.generate_entropy_salt()
        salt = mutate_salt(salt)

    salt_int = core.salt_to_int(salt)
    salt_words = core.salt_to_words(salt, NUM_WORDS)
    state = core.initialize_state(salt_words, NUM_WORDS)

    # Character-level mixing
    for ch in text:
        val = ord(ch)
        for i in range(NUM_WORDS):
            state[i] ^= core.rotate_left(val ^ (i * 31), i)

    # Memory matrix
    matrix = np.zeros((MEM_MATRIX_ROWS, MEM_MATRIX_COLS), dtype=np.uint64)
    memory_matrix = _process_matrix(
        matrix, np.uint64(salt_int),
        np.array(state, dtype=np.uint64)
    )

    # Layer 2: Memory-influenced state
    for r in range(128):
        idx1 = (r * core.MIX_PRIME_1) % MEM_MATRIX_ROWS
        idx2 = (salt_int >> (r % 8)) % MEM_MATRIX_COLS
        mix_val = int(memory_matrix[idx1, idx2])
        for i in range(NUM_WORDS):
            val = (np.uint64(state[i]) ^ np.uint64(mix_val))
            val = val * np.uint64(core.MULT_CONST) + np.uint64(core.INIT_XOR)
            state[i] = int(val) & 0xFFFFFFFFFFFFFFFF

    # Layer 3: Multi-lane processing (4 lanes)
    lane_state = [core.initialize_state(
        [(sw >> i) ^ core.INIT_XOR for sw in salt_words], NUM_WORDS
    ) for i in range(4)]

    for lane_idx in range(4):
        for r in range(16):
            lane_state[lane_idx] = core.arx_round(
                lane_state[lane_idx],
                salt_int ^ (lane_idx * r),
                NUM_WORDS
            )
            for j in range(NUM_WORDS):
                lane_state[lane_idx][j] ^= (state[j] ^ lane_idx ^ r)

    # Merge lanes
    for i in range(NUM_WORDS):
        for lane in lane_state:
            state[i] ^= lane[i]

    # Block processing
    for block in core.split_blocks(padded):
        words = core.words_from_block(block)
        words = [(w ^ salt_int) for w in words]
        words = core.mix_schedule(words)
        for r in range(ROUND_COUNT):
            words = core.arx_round(words, salt_int ^ r, NUM_WORDS)
        for i in range(NUM_WORDS):
            state[i] ^= words[i % NUM_WORDS]

    # Finalize
    state = core.finalize_state(state, salt_words, FINALIZE_ROUNDS, NUM_WORDS)

    # Symbolic output
    hash_result = core.generate_symbolic_output(state, desired_length)
    return f"{VERSION_TAG}.{hash_result}", salt


def verify_password(password: str, stored_hash: str,
                    stored_salt_hex: str) -> bool:
    """Verify password against stored hash."""
    salt_bytes = bytes.fromhex(stored_salt_hex)
    recomputed_hash, _ = aelvonlock512_hash(password, salt=salt_bytes)
    return core.constant_time_compare(recomputed_hash, stored_hash)
