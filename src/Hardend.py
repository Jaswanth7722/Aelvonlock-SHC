"""
Aelvonlock-SHC Hardened
========================
High-security hashing variant for servers and cloud APIs.
Uses ~280 MB memory and 32 ARX rounds with multi-lane processing.

Security: High (suitable for server-side password hashing)
Focus: Strong security with moderate resource usage
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from src import core

# ---------------- Constants ---------------- 
VERSION_TAG = "V.H.L.1"
NUM_WORDS = core.NUM_WORDS  # 8
ROUND_COUNT = 32
FINALIZE_ROUNDS = 5

MEM_MATRIX_ROWS = 6060
MEM_MATRIX_COLS = 6060
NUM_ROUNDS = 256


# ---------------- Numba-Accelerated Operations ----------------

@njit(cache=True)
def _arx_round_scalar(word, key):
    """ARX round for a single 64-bit word."""
    w = np.uint64(word)
    k = np.uint64(key)
    for i in range(4):
        w = w ^ ((k + np.uint64(i * 13)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        w = (w + ((k ^ np.uint64(i * 7)) & np.uint64(0xFFFFFFFFFFFFFFFF))) & np.uint64(0xFFFFFFFFFFFFFFFF)
        w = ((w << np.uint64(5)) | (w >> np.uint64(59))) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return w


@njit(cache=True)
def _process_matrix(matrix, salt_int, state):
    rows, cols = matrix.shape
    n_state = len(state)

    # Fill
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = np.uint64((i * j ^ salt_int) & 0xFFFFFFFFFFFFFFFF)

    # Heavy mixing
    for _ in range(16):
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] ^= matrix[i, j] ^ state[i % n_state] ^ np.uint64(991)

    # ARX diffusion
    for rnd in range(4):
        for i in range(rows):
            for j in range(cols):
                idx2 = (j + rnd) % cols
                matrix[i, j] ^= matrix[i, idx2] ^ state[(i + j + rnd) % n_state]

    return matrix


# ---------------- Main Hashing ----------------

def aelvonlock512_hash(text: str, salt: Optional[bytes] = None,
                       desired_length: int = 64) -> Tuple[str, bytes]:
    """Aelvonlock Hardened hash function."""
    text = core.sanitize_input(text)
    encoded = core.encode_symbols(text)
    binary = core.to_binary(encoded)
    padded = core.pad_binary(binary)

    if salt is None:
        salt = core.generate_entropy_salt()
    salt_int = core.salt_to_int(salt)
    salt_words = core.salt_to_words(salt, NUM_WORDS)

    # Salt mutation
    mutated_salt = bytearray(salt)
    for i in range(len(mutated_salt)):
        mutated_salt[i] ^= ((i * 13 + salt_int % 251) % 256)

    state = core.initialize_state(salt_words, NUM_WORDS)
    state_np = np.array(state, dtype=np.uint64)

    # Layer 1: Large memory matrix
    memory_matrix = np.zeros((MEM_MATRIX_ROWS, MEM_MATRIX_COLS), dtype=np.uint64)
    _process_matrix(memory_matrix, salt_int, state_np)

    # Layer 2: Memory-influenced state mixing
    for r in range(NUM_ROUNDS):
        idx1 = (r * core.MIX_PRIME_1) % MEM_MATRIX_ROWS
        idx2 = (salt_int >> (r % 8)) % MEM_MATRIX_COLS
        mix_val = int(memory_matrix[idx1, idx2])
        for i in range(NUM_WORDS):
            state[i] = ((state[i] ^ mix_val) * core.MULT_CONST + core.INIT_XOR) & 0xFFFFFFFFFFFFFFFF

    # Layer 3: Multi-lane processing
    lane_state = np.zeros(8, dtype=np.uint64)
    for lane in range(4):
        for r in range(16):
            lane_state[lane] = np.uint64(
                _arx_round_scalar(lane_state[lane], salt_int ^ (lane * r))
            )
            for j in range(NUM_WORDS):
                lane_state[lane] ^= np.uint64(state[j] ^ lane ^ r)

    # Merge lanes
    for i in range(NUM_WORDS):
        for lane in lane_state:
            state[i] ^= int(lane)

    # Block processing
    for block in core.split_blocks(padded):
        words = core.words_from_block(block)
        words = [(w ^ salt_int) for w in words]
        words = core.mix_schedule(words)
        for r in range(ROUND_COUNT):
            for i in range(len(words)):
                words[i] = int(_arx_round_scalar(np.uint64(words[i]), np.uint64(salt_int ^ r)))
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
