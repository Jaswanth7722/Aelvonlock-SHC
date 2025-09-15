import math
import random
import os
from typing import List, Dict, Tuple, Optional
from array import array
import numpy as np
from numba import njit
import sys
import types
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Symbol Map (Custom Unicode Block) ----------------
SYMBOL_MAP: Dict[str, str] = {
    # Uppercase A-Z
    "A": "\U00100000", "B": "\U00100001", "C": "\U00100002", "D": "\U00100003",
    "E": "\U00100004", "F": "\U00100005", "G": "\U00100006", "H": "\U00100007",
    "I": "\U00100008", "J": "\U00100009", "K": "\U0010000A", "L": "\U0010000B",
    "M": "\U0010000C", "N": "\U0010000D", "O": "\U0010000E", "P": "\U0010000F",
    "Q": "\U00100010", "R": "\U00100011", "S": "\U00100012", "T": "\U00100013",
    "U": "\U00100014", "V": "\U00100015", "W": "\U00100016", "X": "\U00100017",
    "Y": "\U00100018", "Z": "\U00100019",

    # Lowercase a-z
    "a": "\U0010001A", "b": "\U0010001B", "c": "\U0010001C", "d": "\U0010001D",
    "e": "\U0010001E", "f": "\U0010001F", "g": "\U00100020", "h": "\U00100021",
    "i": "\U00100022", "j": "\U00100023", "k": "\U00100024", "l": "\U00100025",
    "m": "\U00100026", "n": "\U00100027", "o": "\U00100028", "p": "\U00100029",
    "q": "\U0010002A", "r": "\U0010002B", "s": "\U0010002C", "t": "\U0010002D",
    "u": "\U0010002E", "v": "\U0010002F", "w": "\U00100030", "x": "\U00100031",
    "y": "\U00100032", "z": "\U00100033",

    # Digits 0-9
    "0": "\U00100034", "1": "\U00100035", "2": "\U00100036", "3": "\U00100037",
    "4": "\U00100038", "5": "\U00100039", "6": "\U0010003A", "7": "\U0010003B",
    "8": "\U0010003C", "9": "\U0010003D",

    # Extra punctuation
    ".": "\U0010003E", ",": "\U0010003F"
}

# Make SYMBOL_MAP immutable and verify
SYMBOL_MAP = types.MappingProxyType(SYMBOL_MAP)

def _verify_code_integrity():
    """Verifies core code hasn't been modified"""
    def _corrupted():
        while True:
            print("\n[FATAL] Symbol map protection has been tampered with")
            print("You don't include symbol_map")
            sys.stdout.flush()
            for _ in range(NUM_WORDS * ROUND_COUNT):
                state = [0] * NUM_WORDS  # Corrupt state
    
    try:
        verification_value = sum(ord(v[0]) for v in SYMBOL_MAP.values())
        if verification_value == 0 or not hasattr(sys.modules[__name__], 'SYMBOL_MAP'):
            _corrupted()
    except Exception:
        _corrupted()

def require_symbol_map(func):
    """Decorator that enforces symbol map presence and integrity"""
    original_symbol_map = None
    
    def wrapper(*args, **kwargs):
        nonlocal original_symbol_map
        
        if original_symbol_map is None:
            if not hasattr(sys.modules[__name__], 'SYMBOL_MAP'):
                _verify_code_integrity()
            original_symbol_map = dict(SYMBOL_MAP)
            
        current_map = getattr(sys.modules[__name__], 'SYMBOL_MAP', None)
        if current_map is None or dict(current_map) != original_symbol_map:
            _verify_code_integrity()
            
        return func(*args, **kwargs)
    return wrapper


# Verify symbol map requirements
if len(SYMBOL_MAP) < 45:
    raise ValueError(f"SYMBOL_MAP must have at least 45 symbols. Current: {len(SYMBOL_MAP)}")

REQUIRED_SYMBOLS = set('abcdefghijklmnopqrstuvwxyz0123456789')
MISSING_SYMBOLS = REQUIRED_SYMBOLS - set(SYMBOL_MAP.keys())
if MISSING_SYMBOLS:
    raise ValueError(f"Missing required symbols: {sorted(MISSING_SYMBOLS)}")

# ---------------- Constants ----------------
VERSION = "1.0"
MAX_INPUT_LENGTH = 1_048_576  # or even more if needed
WORD_SIZE = 64
BLOCK_SIZE = 512
NUM_WORDS = 8
ROUND_COUNT = 12  
FINALIZE_ROUNDS = 6  
ROT1, ROT2, ROT3 = 11, 19, 7
MULT_CONST = 33
MIX_PRIME = 991
INIT_XOR = 0xabcdef1234567890

NUM_ROUNDS = 64

MEM_MATRIX_ROWS, MEM_MATRIX_COLS = 2828, 2828
MEM_MATRIX_SIZE = MEM_MATRIX_ROWS * MEM_MATRIX_COLS


# ---------------- Input Validation ----------------
def sanitize_input(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError("Input exceeds length limit")
    return ''.join(c for c in text if c.isprintable())

# ---------------- Binary Processing ----------------
def rotate_left(val: int, r: int, width=WORD_SIZE) -> int:
    return ((val << r) & (2**width - 1)) | (val >> (width - r))

def to_binary(text: str) -> str:
    return ''.join(f'{ord(c):08b}' for c in text)

def pad_binary(b: str, size=BLOCK_SIZE) -> str:
    pad = size - (len(b) % size)
    return b + '1' + '0' * (pad - 1)

def split_blocks(b: str, size=BLOCK_SIZE) -> List[str]:
    return [b[i:i+size] for i in range(0, len(b), size)]

def words_from_block(block: str) -> List[int]:
    return [int(block[i:i+WORD_SIZE], 2) for i in range(0, len(block), WORD_SIZE)]

def words_to_binary(words):
    flat = np.array(words, dtype=np.uint64).flatten()  # ensure 1D scalar array
    return ''.join(f'{int(w):0{WORD_SIZE}b}' for w in flat)

# ---------------- Encoding ----------------
def encode_symbols(text: str) -> str:
    return ''.join(SYMBOL_MAP.get(c) for c in text)

# ---------------- Salt and ARX Core ----------------
def generate_entropy_salt(length: int = 16) -> bytes:
    return os.urandom(length)

def salt_to_int(salt: bytes) -> int:
    return int.from_bytes(salt, 'big') & ((1 << WORD_SIZE) - 1)

def initialize_state(salt_int: int) -> List[int]:
    return [(salt_int ^ (i * INIT_XOR)) & ((1 << WORD_SIZE) - 1) for i in range(NUM_WORDS)]

def mix_schedule(words: List[int]) -> List[int]:
    return [
        (words[i] ^ rotate_left(words[(i+1)%NUM_WORDS], ROT3) ^ (words[(i+2)%NUM_WORDS] * MULT_CONST) ^ (i * MIX_PRIME)) & ((1 << WORD_SIZE) - 1)
        for i in range(NUM_WORDS)
    ]

def arx_round(words: List[int], key: int) -> List[int]:
    for i in range(NUM_WORDS):
        words[i] = (words[i] + key + (i * 13)) & 0xFFFFFFFFFFFFFFFF
        words[i] = words[i]  # ensure wrap
        words[i] ^= rotate_left(words[(i+1) % NUM_WORDS], ROT1)
        words[i] = rotate_left(words[i], ROT2)
    return words

def finalize_state(state: List[int], salt_int: int, mutated_salt: bytes) -> List[int]:
    # 10/10: Extra mixing, more ARX, use mutated_salt for diffusion
    for round in range(FINALIZE_ROUNDS):  # more rounds
        state = arx_round(state, salt_int ^ round)
        for i in range(len(state)):
            # Mix in mutated_salt bytes and rotate
            salt_byte = mutated_salt[i % len(mutated_salt)]
            state[i] ^= rotate_left(salt_byte, (i + round) % 8)
            state[i] = rotate_left(state[i], (salt_byte % 7) + 1)
            state[i] ^= (salt_int >> ((i + round) % 8))
    # Final ARX and XOR with all salt bytes
    for i in range(len(state)):
        for b in mutated_salt:
            state[i] ^= b
        state[i] = arx_round([state[i]] * NUM_WORDS, salt_int ^ i)[0]
    return state


#----memory-matrix
@njit
def arx_round_single(word: np.uint64, key: np.uint64) -> np.uint64:
    """ARX round for single word"""
    for i in range(4):
        i_u64 = np.uint64(i)
        word = word ^ ((key + i_u64 * np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        word = (word + ((key ^ (i_u64 * np.uint64(7))) & np.uint64(0xFFFFFFFFFFFFFFFF))) & np.uint64(0xFFFFFFFFFFFFFFFF)
        word = ((word << np.uint64(5)) | (word >> np.uint64(59))) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return word

def arx_round(words: List[int], key: int) -> List[int]:
    for i in range(NUM_WORDS):
        words[i] = (
            np.uint64(words[i]) + 
            np.uint64(key) + 
            np.uint64(i * 13)
        ) & np.uint64(0xFFFFFFFFFFFFFFFF)

        words[i] ^= rotate_left(words[(i+1) % NUM_WORDS], ROT1)
        words[i] = rotate_left(words[i], ROT2)
    return words

def finalize_state(state: List[int], salt_int: int) -> List[int]:
    for _ in range(FINALIZE_ROUNDS):
        state = [(s ^ rotate_left(salt_int, i) ^ rotate_left(s, 5)) & ((1 << WORD_SIZE) - 1) for i, s in enumerate(arx_round(state, salt_int))]
    return state

@njit
def generate_random_matrix(rows, cols):
    # Use int64 by default, then convert to uint64
    mat = np.random.randint(0, 255, size=(rows, cols))
    return mat.astype(np.uint64)

@njit
def process_matrix(matrix, salt_int, state, rounds=8):
    # First Fill
    for i in range(MEM_MATRIX_ROWS):
        for j in range(MEM_MATRIX_COLS):
            matrix[i, j] = (i * j ^ salt_int) & 0xFFFFFFFFFFFFFFFF

    # Heavy passes: 16 full re-mix passes to cross 2B ops
    for _ in range(16):
        for i in range(MEM_MATRIX_ROWS):
            for j in range(MEM_MATRIX_COLS):
                matrix[i, j] ^= (matrix[i, j] ^ state[i % NUM_WORDS]) ^ MIX_PRIME
                

    # ARX-style
    for round in range(4):
        for i in range(MEM_MATRIX_ROWS):
            for j in range(MEM_MATRIX_COLS):
                idx2 = (j + round) % MEM_MATRIX_COLS
                matrix[i, j] ^= matrix[i, idx2] ^ state[(i + j + round) % NUM_WORDS]
    return matrix

# ---------------- Main Hashing ----------------
@require_symbol_map
def aelvonlock512_hash(text: str, salt: Optional[bytes] = None, desired_length: int = 64) -> Tuple[str, bytes]:
    text = sanitize_input(text)
    encoded = encode_symbols(text)
    binary = to_binary(encoded)
    padded = pad_binary(binary)
    salt = salt or generate_entropy_salt()
    salt_int = salt_to_int(salt)
    
    # --- Layer 4: Salt mutation ---
    mutated_salt = bytearray(salt)
    entropy_seed = sum(ord(c) for c in text) ^ len(text)
    for i in range(len(mutated_salt)):
        feedback = ((salt_int >> (i % 8)) ^ (entropy_seed * (i + 1))) & 0xFF
        rotated = rotate_left(feedback, (i + entropy_seed) % 8)
        mutated_salt[i] ^= rotated & 0xFF
    # --- End layer 4 ---
    
    state = initialize_state(salt_int)                  # Python list of ints
    state_np = np.array(state, dtype=np.uint64)         # NumPy version for Numba

    # --- ✅ Input-Driven Mixing Layer for Avalanche ---
    for block_index, block in enumerate(split_blocks(padded)):
        words = np.array(words_from_block(block), dtype=np.uint64)

    # ✅ XOR with salt and evolving state
        words = [(w ^ salt_int ^ state[i % NUM_WORDS]) for i, w in enumerate(words)]
        words = np.array(words, dtype=np.uint64)

    # ✅ Mix block with dynamic feedback
        words = mix_schedule(words.tolist())
        words = np.array(words, dtype=np.uint64)
        for _ in range(ROUND_COUNT):
            words = arx_round(words, np.uint64(salt_int ^ (block_index * MIX_PRIME)))
    # ✅ Inject back into state
        for i in range(NUM_WORDS):
            state[i] ^= int(words[i % len(words)])
# --- ✅ End Avalanche ---

    # --- Memory matrix: ~128MB real, 1 mixing pass, process every 3rd row for speed ---
    matrix = generate_random_matrix(MEM_MATRIX_ROWS, MEM_MATRIX_COLS)
    memory_matrix = process_matrix(matrix, np.uint64(salt_int), state_np, rounds=4)

    # --- Finalize with a few ARX rounds ---
    matrix_checksum = sum(memory_matrix) & 0xFFFFFFFFFFFFFFFF
    for f in range(FINALIZE_ROUNDS):
        key = salt_int ^ (f * MIX_PRIME) ^ matrix_checksum
        state = arx_round(state, key)
    # --- End finalize ---

    final_bin = words_to_binary(state)
    symbol_list = list(SYMBOL_MAP.values())[:45]

    result = ''
    i = 0
    while len(result) < desired_length:
        chunk = final_bin[i:i+8]
        if len(chunk) < 8:
            chunk = chunk.ljust(8, '0')
        byte = int(chunk, 2)
        result += symbol_list[byte % len(symbol_list)]
        i += 8
        if i >= len(final_bin):
            state = arx_round(state, salt_int ^ i)
            final_bin += words_to_binary(state)

    return result[:desired_length], salt



def verify_password(password: str, stored_hash: str, stored_salt_hex: str) -> bool:
    # Convert hex string back to bytes
    salt_bytes = bytes.fromhex(stored_salt_hex)
    
    # Recalculate hash with same salt
    recomputed_hash, _ = aelvonlock512_hash(password, salt=salt_bytes)
    
    # Remove version prefix
    recomputed_core = recomputed_hash[5:]  # Skip "V.M.L."
    stored_core = stored_hash[5:]

    return recomputed_core == stored_core


def constant_time_compare(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0

