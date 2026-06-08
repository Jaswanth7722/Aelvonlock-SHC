"""
Aelvonlock-SHC Fast
====================
Ultra-fast hashing variant for embedded systems and constrained devices.
Uses minimal memory (~64 MB) and very few ARX rounds (4).

Security: Basic (not suitable for high-security applications)
Focus: Maximum speed, minimal resource usage
"""

from typing import List, Optional, Tuple

from src import core

# ---------------- Constants ----------------
VERSION_TAG = "V.F.S.1"
NUM_WORDS = 4
ROUND_COUNT = 4
FINALIZE_ROUNDS = 4

MEM_MATRIX_ROWS = 2000
MEM_MATRIX_COLS = 2000
MEM_MATRIX_SIZE = MEM_MATRIX_ROWS * MEM_MATRIX_COLS


# ---------------- Main Hashing ----------------

def aelvonlock512_hash(text: str, salt: Optional[bytes] = None,
                       desired_length: int = 64) -> Tuple[str, bytes]:
    """Aelvonlock Fast hash function - maximum speed."""
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
        words = core.words_from_block(block)
        words = [(w ^ salt_int ^ state[i % NUM_WORDS]) for i, w in enumerate(words)]
        words = core.mix_schedule(words)
        for _ in range(ROUND_COUNT):
            words = core.arx_round(words, salt_int ^ (block_index * core.MIX_PRIME_1))
        for i in range(NUM_WORDS):
            state[i] ^= words[i % len(words)]

    # Memory matrix (sparse)
    mem = [0] * MEM_MATRIX_SIZE
    for i in range(0, MEM_MATRIX_ROWS, 64):
        base = i * MEM_MATRIX_COLS
        for j in range(MEM_MATRIX_COLS):
            mem[base + j] = (i * j ^ salt_int) & 0xFFFFFFFFFFFFFFFF

    for mix_pass in range(1):
        for i in range(0, MEM_MATRIX_ROWS, 512):
            base = i * MEM_MATRIX_COLS
            for j in range(MEM_MATRIX_COLS):
                mem[base + j] ^= (state[j % NUM_WORDS] ^ (mix_pass * core.MIX_PRIME_1))
            for j in range(MEM_MATRIX_COLS):
                state[j % NUM_WORDS] ^= mem[base + j]

    # Finalize
    for _ in range(FINALIZE_ROUNDS):
        state = core.arx_round(state, salt_int)

    # Symbolic output
    hash_result = core.generate_symbolic_output(state, desired_length)
    return f"{VERSION_TAG}.{hash_result}", salt


def verify_password(password: str, stored_hash: str,
                    stored_salt_hex: str) -> bool:
    """Verify password against stored hash."""
    salt_bytes = bytes.fromhex(stored_salt_hex)
    recomputed_hash, _ = aelvonlock512_hash(password, salt=salt_bytes)
    return core.constant_time_compare(recomputed_hash, stored_hash)
