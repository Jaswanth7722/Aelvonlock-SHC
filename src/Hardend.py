import math
import random
import os
from typing import List, Dict, Tuple, Optional
from array import array
from numba import njit
import sys
import types
import numpy as np
import warnings

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
VERSION_SYMBOL = "V.H.L.1"
WORD_SIZE = 64
BLOCK_SIZE = 512
NUM_WORDS = 8
ROUND_COUNT = 32
FINALIZE_ROUNDS = 5
ROT1, ROT2, ROT3 = 11, 19, 7
MULT_CONST = 33
MIX_PRIME = 991
INIT_XOR = 0xabcdef1234567890

NUM_ROUNDS = 256
MEM_MATRIX_ROWS = 6060
MEM_MATRIX_COLS = 6060
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

def words_to_binary(words: List[int]) -> str:
    return ''.join(f'{w:0{WORD_SIZE}b}' for w in words)

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

@njit
def arx_round(words, key):
    # words: np.ndarray of uint64
    key = np.uint64(key)
    for i in range(words.size):
        words[i] = (words[i] + key + np.uint64(i*13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        words[i] ^= np.uint64((words[(i+1)%words.size] << ROT1) | (words[(i+1)%words.size] >> (64-ROT1)))
        words[i] = np.uint64((words[i] << ROT2) | (words[i] >> (64-ROT2)))
    return words

def finalize_state(state: List[int], salt_int: int) -> List[int]:
    for _ in range(FINALIZE_ROUNDS):
        arx_result = [arx_round_scalar(np.uint64(s), np.uint64(salt_int)) for s in state]
    state = [(s ^ rotate_left(salt_int, i) ^ rotate_left(s, 5)) & ((1 << WORD_SIZE) - 1) for i, s in enumerate(arx_result)]

    return state

@njit
def process_matrix(matrix, salt_int, state, rounds=8):
    for i in range(MEM_MATRIX_ROWS):
        for j in range(MEM_MATRIX_COLS):
            matrix[i, j] = (i * j ^ salt_int) & 0xFFFFFFFFFFFFFFFF

    for _ in range(16):
        for i in range(MEM_MATRIX_ROWS):
            for j in range(MEM_MATRIX_COLS):
                matrix[i, j] ^= (matrix[i, j] ^ state[i % NUM_WORDS]) ^ MIX_PRIME

    for round in range(4):
        for i in range(MEM_MATRIX_ROWS):
            for j in range(MEM_MATRIX_COLS):
                idx2 = (j + round) % MEM_MATRIX_COLS
                matrix[i, j] ^= matrix[i, idx2] ^ state[(i + j + round) % NUM_WORDS]

    return matrix

@njit
def arx_round_single(word: np.uint64, key: np.uint64) -> np.uint64:
    """ARX round for a single 64-bit word"""
    for i in range(4):
        word = word ^ ((key + np.uint64(i*13)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        word = (word + ((key ^ np.uint64(i*7)) & np.uint64(0xFFFFFFFFFFFFFFFF))) & np.uint64(0xFFFFFFFFFFFFFFFF)
        word = ((word << np.uint64(5)) | (word >> np.uint64(59))) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return word
@njit
def arx_round_scalar(word: np.uint64, key: np.uint64) -> np.uint64:
    """ARX round for a single 64-bit word"""
    for i in range(4):
        word = word ^ ((key + np.uint64(i*13)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        word = (word + ((key ^ np.uint64(i*7)) & np.uint64(0xFFFFFFFFFFFFFFFF))) & np.uint64(0xFFFFFFFFFFFFFFFF)
        word = ((word << np.uint64(5)) | (word >> np.uint64(59))) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return word


# ---------------- Main Hashing ----------------
@require_symbol_map
def aelvonlock512_hash(text: str, salt: Optional[bytes] = None, desired_length: int = 64) -> Tuple[str, bytes]:
    text = sanitize_input(text)
    encoded = encode_symbols(text)
    binary = to_binary(encoded)
    padded = pad_binary(binary)
    if salt is None:
        salt = generate_entropy_salt()
    salt_int = salt_to_int(salt)
    
    # --- Layer 4: Salt mutation ---
    mutated_salt = bytearray(salt)
    for i in range(len(mutated_salt)):
        mutated_salt[i] ^= ((i * 13 + salt_int % 251) % 256)
    # --- End layer 4 ---
    
    state = initialize_state(salt_int)

    # --- Extra Layer: Large memory matrix and scrambling ---
    memory_matrix = np.zeros((MEM_MATRIX_ROWS, MEM_MATRIX_COLS), dtype=np.uint64)
    state_np = np.array(state, dtype=np.uint64)
    process_matrix(memory_matrix, salt_int, state_np)
    

# --- End extra layer ---

    # --- End extra layer ---

    # --- Layer 2: Use memory to influence hashing state ---

    for r in range(NUM_ROUNDS):
        idx1 = (r * MIX_PRIME) % MEM_MATRIX_ROWS
        idx2 = (salt_int >> (r % 8)) % MEM_MATRIX_COLS
        mix_val = memory_matrix[idx1, idx2]
        for i in range(NUM_WORDS):
            state[i] = ((state[i] ^ mix_val) * MULT_CONST + INIT_XOR) & 0xFFFFFFFFFFFFFFFF
# --- End layer 2 ---

    # --- End layer 2 ---

    # --- Layer 3: Multiple lane processing and merging ---
    lane_state = np.zeros(8, dtype=np.uint64)
    for i in range(4):
        for r in range(16):  # fewer rounds than main
            lane_state[i] = arx_round_single(
            lane_state[i],
            np.uint64(salt_int ^ (i * r))
            )


            for j in range(NUM_WORDS):
                lane_state[i] ^= (state[j] ^ i ^ r)

    # Merge all lanes back
    for i in range(NUM_WORDS):
        for lane in lane_state:
            state[i] ^= lane
    # --- End layer 3 ---

    for block in split_blocks(padded):
        words = words_from_block(block)
        words = [(w ^ salt_int) for w in words]
        words = mix_schedule(words)
        for r in range(ROUND_COUNT):
            for i in range(len(words)):
                words[i] = arx_round_scalar(np.uint64(words[i]), np.uint64(salt_int ^ r))


        for i in range(NUM_WORDS):
            state[i] ^= words[i % NUM_WORDS]

    state = finalize_state(state, salt_int)
    final_bin = words_to_binary(state)
    symbol_list = list(SYMBOL_MAP.values())

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
            for j in range(len(state)):
                state[j] = arx_round_scalar(np.uint64(state[j]), np.uint64(salt_int ^ i))


            final_bin += words_to_binary(state)

    return VERSION_SYMBOL + result[:desired_length], salt

def verify_password(password: str, stored_hash: str, stored_salt_hex: str) -> bool:
    recomputed_hash, _ = aelvonlock512_hash(password, salt=bytes.fromhex(stored_salt_hex))
    
    if len(recomputed_hash) != len(stored_hash):
        return False
    
    result = 0
    for x, y in zip(recomputed_hash[5:], stored_hash[5:]):  # Skip version prefix
        result |= ord(x) ^ ord(y)
    return result == 0
