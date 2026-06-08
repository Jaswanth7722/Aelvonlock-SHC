"""
Aelvonlock-SHC Balanced
========================
Optimized for both performance and security.
Uses moderate memory (~128 MB) and ARX rounds (12).

Security: Standard (good for general-purpose applications)
Focus: Balanced tradeoff between speed and security
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from src import core

# ---------------- Constants ---------------- 
VERSION_TAG = "V.B.L.1"
NUM_WORDS = core.NUM_WORDS  # 8
ROUND_COUNT = 12
FINALIZE_ROUNDS = 6

MEM_MATRIX_ROWS = 2828
MEM_MATRIX_COLS = 2828
MEM_MATRIX_SIZE = MEM_MATRIX_ROWS * MEM_MATRIX_COLS


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
                matrix[i, j] ^= matrix[i, j] ^ state[i % n_state] ^ np.uint64(991)

    # ARX-style
    for rnd in range(4):
        for i in range(rows):
            for j in range(cols):
                idx2 = (j + rnd) % cols
                matrix[i, j] ^= matrix[i, idx2] ^ state[(i + j + rnd) % n_state]

    return matrix


# ---------------- Main Hashing ----------------

def aelvonlock512_hash(text: str, salt: Optional[bytes] = None,
                       desired_length: int = 64) -> Tuple[str, bytes]:
    """Aelvonlock Balanced hash function."""
    text = core.sanitize_input(text)
    encoded = core.encode_symbols(text)
    binary = core.to_binary(encoded)
    padded = core.pad_binary(binary)
    salt = salt or core.generate_entropy_salt()
    salt_int = core.salt_to_int(salt)
    salt_words = core.salt_to_words(salt, NUM_WORDS)

    # Salt mutation
    mutated_salt = bytearray(salt)
    entropy_seed = sum(ord(c) for c in text) ^ len(text)
    for i in range(len(mutated_salt)):
        feedback = ((salt_int >> (i % 8)) ^ (entropy_seed * (i + 1))) & 0xFF
        rotated = core.rotate_left(feedback, (i + entropy_seed) % 8)
        mutated_salt[i] ^= rotated & 0xFF

    state = core.initialize_state(salt_words, NUM_WORDS)
    state_np = np.array(state, dtype=np.uint64)

    # Input-driven mixing
    for block_index, block in enumerate(core.split_blocks(padded)):
        words_list = core.words_from_block(block)
        words_list = [(w ^ salt_int ^ state[i % NUM_WORDS])
                      for i, w in enumerate(words_list)]
        words_np = np.array(words_list, dtype=np.uint64)
        words_list = core.mix_schedule(words_list)
        words_np = np.array(words_list, dtype=np.uint64)

        for _ in range(ROUND_COUNT):
            words_np = np.array(
                core.arx_round(words_np.tolist(),
                               salt_int ^ (block_index * core.MIX_PRIME_1)),
                dtype=np.uint64
            )

        for i in range(NUM_WORDS):
            state[i] ^= int(words_np[i % len(words_np)])

    # Memory matrix
    matrix = np.zeros((MEM_MATRIX_ROWS, MEM_MATRIX_COLS), dtype=np.uint64)
    memory_matrix = _process_matrix(matrix, np.uint64(salt_int), state_np)

    # Finalize
    matrix_checksum = int(np.sum(memory_matrix)) & 0xFFFFFFFFFFFFFFFF
    for f in range(FINALIZE_ROUNDS):
        key = salt_int ^ (f * core.MIX_PRIME_1) ^ matrix_checksum
        state = core.arx_round(state, key, NUM_WORDS)

    # Symbolic output
    hash_result = core.generate_symbolic_output(state, desired_length)
    return f"{VERSION_TAG}.{hash_result}", salt


def verify_password(password: str, stored_hash: str,
                    stored_salt_hex: str) -> bool:
    """Verify password against stored hash."""
    salt_bytes = bytes.fromhex(stored_salt_hex)
    recomputed_hash, _ = aelvonlock512_hash(password, salt=salt_bytes)
    return core.constant_time_compare(recomputed_hash, stored_hash)
