"""
Aelvonlock-SHC Lite
===================
Lightweight hashing variant for low-resource environments.
Uses minimal memory (~64 MB) and few ARX rounds (4).

Security: Basic (not suitable for high-security applications)
Focus: Speed and minimal resource usage over security
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from src import core

# ---------------- Constants ----------------
VERSION_TAG = "V.L.T.1"
NUM_WORDS = 4
ROUND_COUNT = 4
FINALIZE_ROUNDS = 4

MEM_MATRIX_ROWS = 2828
MEM_MATRIX_COLS = 2828


# ---------------- Numba-Accelerated Operations ----------------

@njit(cache=True)
def _arx_round_numba(words, key):
    key = np.uint64(key)
    n = len(words)
    for i in range(n):
        words[i] = (words[i] + key + np.uint64(i * 13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        words[i] ^= np.uint64((words[(i + 1) % n] << 11) | (words[(i + 1) % n] >> 53))
        words[i] = np.uint64((words[i] << 19) | (words[i] >> 45))
    return words


# ---------------- Main Hashing ----------------

def aelvonlock512_hash(text: str, salt: Optional[bytes] = None,
                       desired_length: int = 64) -> Tuple[str, bytes]:
    """Aelvonlock Lite hash function."""
    text = core.sanitize_input(text)
    encoded = core.encode_symbols(text)
    binary = core.to_binary(encoded)
    padded = core.pad_binary(binary)
    salt = salt or core.generate_entropy_salt()
    salt_int = core.salt_to_int(salt)

    # Salt mutation
    mutated_salt = bytearray(salt)
    for i in range(len(mutated_salt)):
        mutated_salt[i] ^= ((i * 13 + salt_int % 251) % 256)

    state = core.initialize_state(
        core.salt_to_words(salt, NUM_WORDS), NUM_WORDS
    )

    # Input-driven mixing
    for block_index, block in enumerate(core.split_blocks(padded)):
        words_list = core.words_from_block(block)
        # XOR with salt and state
        words_list = [(w ^ salt_int ^ state[i % NUM_WORDS])
                      for i, w in enumerate(words_list)]
        words_np = np.array(words_list, dtype=np.uint64)

        for _ in range(ROUND_COUNT):
            words_np = _arx_round_numba(
                words_np, np.uint64(salt_int ^ (block_index * core.MIX_PRIME_1))
            )

        # Inject into state
        for i in range(NUM_WORDS):
            state[i] ^= int(words_np[i % len(words_np)])

    # Finalize
    state_np = np.array(state, dtype=np.uint64)
    for _ in range(FINALIZE_ROUNDS):
        state_np = _arx_round_numba(state_np, np.uint64(salt_int))

    # Symbolic output
    hash_result = core.generate_symbolic_output(state_np.tolist(), desired_length)
    return f"{VERSION_TAG}.{hash_result}", salt


def verify_password(password: str, stored_hash: str,
                    stored_salt_hex: str) -> bool:
    """Verify password against stored hash."""
    salt_bytes = bytes.fromhex(stored_salt_hex)
    recomputed_hash, _ = aelvonlock512_hash(password, salt=salt_bytes)
    return core.constant_time_compare(recomputed_hash, stored_hash)
