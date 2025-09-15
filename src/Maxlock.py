import math
import random
import os
from typing import List, Dict, Tuple, Optional
from array import array
import numpy as np
from numba import njit, prange
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


SYMBOL_MAP = types.MappingProxyType(SYMBOL_MAP)
def require_symbol_map(func):
    """Decorator that enforces symbol map presence and integrity"""
    original_symbol_map = SYMBOL_MAP.copy()  # Store original state
    
    def wrapper(*args, **kwargs):
        # Check if SYMBOL_MAP exists and matches original
        if not hasattr(sys.modules[__name__], 'SYMBOL_MAP'):
            print("\n[FATAL] You don't include symbol_map")
            sys.exit(1)
            
        current_map = getattr(sys.modules[__name__], 'SYMBOL_MAP')
        if current_map != original_symbol_map:
            print("\n[FATAL] Symbol map has been modified")
            sys.exit(1)
            
        return func(*args, **kwargs)
        
    return wrapper

# ---------------- Constants ----------------
VERSION = "1.0"
MAX_INPUT_LENGTH = 1_048_576  # or even more if needed
WORD_SIZE = 64
BLOCK_SIZE = 512
NUM_WORDS = 8
ROUND_COUNT = 32
FINALIZE_ROUNDS = 5
ROT1, ROT2, ROT3 = 11, 19, 7
MULT_CONST = 33
MIX_PRIME = 991
INIT_XOR = 0xabcdef1234567890

NUM_ROUNDS = 128
VERSION_SYMBOL = "V.M.L.1"

MEM_MATRIX_ROWS = 8192
MEM_MATRIX_COLS = 16384
MEM_MATRIX_SIZE = MEM_MATRIX_ROWS * MEM_MATRIX_COLS

# ---------------- Input Validation ----------------
def sanitize_input(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError("Input exceeds length limit")
    return ''.join(c for c in text if c.isprintable())

if len(SYMBOL_MAP) < 32 or len(SYMBOL_MAP) > 128:
    raise ValueError("Symbol list must contain between 32 and 128 symbols.")

# ---------------- Binary Processing ----------------
def rotate_left(val: int, r: int, width=WORD_SIZE) -> int:
    return ((val << r) & (2**width - 1)) | (val >> (width - r))

def to_binary(text: str) -> str:
    return ''.join(f'{ord(c):08b}' for c in text)
def pad_binary(b: str, size=BLOCK_SIZE) -> str:
    pad_len = (size - (len(b) + 9) % size) % size  # 1 for '1', 8 for pad_info
    pad_info = f'{pad_len:08b}'
    return b + '1' + '0' * pad_len + pad_info

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
    return int.from_bytes(salt[:8], 'big') & ((1 << WORD_SIZE) - 1)

def initialize_state(salt_int: int) -> List[int]:
    return [(salt_int ^ (i * INIT_XOR)) & ((1 << WORD_SIZE) - 1) for i in range(NUM_WORDS)]

def mix_schedule(words: List[int]) -> List[int]:
    if len(words) < NUM_WORDS:
        raise ValueError("Words list is too short for mix schedule.")
    
    return [
        (words[i] ^ rotate_left(words[(i+1)%NUM_WORDS], ROT3) ^ (words[(i+2)%NUM_WORDS] * MULT_CONST) ^ (i * MIX_PRIME)) & ((1 << WORD_SIZE) - 1)
        for i in range(NUM_WORDS)
    ]

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
@require_symbol_map  # Add this decorator
def aelvonlock512_hash(text: str, salt: Optional[bytes] = None, desired_length: int = 64) -> Tuple[str, bytes]:
    text = sanitize_input(text)
    encoded = encode_symbols(text)
    binary = to_binary(encoded)
    padded = pad_binary(binary)

    
    # --- Layer 4: Salt mutation ---
    def mutate_salt(salt: bytes) -> bytes:
        pad = len(salt)
        entropy = array('B', [0] * pad)
        for i in range(pad):
            entropy[i] = ((salt[i] * (i + 7)) ^ ((salt[(i - 1) % pad] + i * 19) & 0xFF)) % 256
        return bytes(entropy)


    # --- End layer 4 ---
    if salt is None:
        salt = generate_entropy_salt()
        salt = mutate_salt(salt)    
    salt_int = salt_to_int(salt)# After converting to int
    state = initialize_state(salt_int)
    for ch in text:
        val = ord(ch)
        for i in range(NUM_WORDS):
            state[i] ^= rotate_left(val ^ (i * 31), i)


    matrix = generate_random_matrix(MEM_MATRIX_ROWS, MEM_MATRIX_COLS)
    memory_matrix = process_matrix(matrix, np.uint64(salt_int), np.array(state, dtype=np.uint64), rounds=4)



    # --- End extra layer ---

    # --- Layer 2: Use memory to influence hashing state ---
    for r in range(128):
        idx1 = (r * MIX_PRIME) % MEM_MATRIX_ROWS
        idx2 = (salt_int >> (r % 8)) % MEM_MATRIX_COLS
        mix_val = memory_matrix[idx1, idx2]
        for i in range(NUM_WORDS):
            state[i] = (
                (np.uint64(state[i]) ^ np.uint64(mix_val)) * 
                np.uint64(MULT_CONST) + 
                np.uint64(INIT_XOR)
            ) & np.uint64(0xFFFFFFFFFFFFFFFF)

    # --- End layer 2 ---

    # --- Layer 3: Multiple lane processing and merging ---
    lane_state = [initialize_state((salt_int >> i) ^ INIT_XOR) for i in range(4)]
    for i in range(4):
        for r in range(16):  # fewer rounds than main
            lane_state[i] = arx_round(lane_state[i], salt_int ^ (i * r))
            for j in range(NUM_WORDS):
                lane_state[i][j] ^= (state[j] ^ i ^ r)

    # Merge all lanes back
    for i in range(NUM_WORDS):
        for lane in lane_state:
            state[i] ^= lane[i]
    # --- End layer 3 ---

    for block in split_blocks(padded):
        words = words_from_block(block)
        words = [(w ^ salt_int) for w in words]
        words = mix_schedule(words)
        for r in range(ROUND_COUNT):
            words = arx_round(words, salt_int ^ r)
        for i in range(NUM_WORDS):
            state[i] ^= words[i % NUM_WORDS]

    state = finalize_state(state, salt_int)
    final_bin = words_to_binary(state)
    symbol_list = list(SYMBOL_MAP.values())
    if len(set(symbol_list)) != len(symbol_list):
        raise ValueError("Symbol map contains duplicate symbols.")


    result = ''
    i = 0
    MAX_OUTPUT_BITS = 131072  # Safety cap: 16KB output
    mod_base = 256 - (256 % len(symbol_list))  # E.g., 225 if 45 symbols
    retries = 0
    MAX_RETRIES = 5000  # Prevent infinite loop

    while len(result) < desired_length and i + 8 <= len(final_bin):
        chunk = final_bin[i:i+8]
        i += 8  # Move to next chunk each loop

        if len(chunk) < 8:
            chunk = chunk.ljust(8, '0')

        byte = int(chunk, 2)

        if len(symbol_list) == 0:
            raise ValueError("Symbol list is empty")


        if byte < mod_base:
            symbol = symbol_list[byte % len(symbol_list)]
            result += symbol
            retries = 0  # reset retries on success
        else:
            retries += 1
            if retries >= MAX_RETRIES:
                # Fallback: Accept biased symbol to move on
                symbol = symbol_list[byte % len(symbol_list)]
                result += symbol
                retries = 0


        i += 8
        if i >= len(final_bin):
            if len(final_bin) > MAX_OUTPUT_BITS:
                raise RuntimeError("Hash generation exceeded safe output bounds")

            state = arx_round(state, salt_int ^ i)
            state = arx_round(state, salt_int ^ i ^ len(result))
            final_bin += words_to_binary(state)


    return VERSION_SYMBOL + result[:desired_length], salt


# ---------------- Verifier ----------------
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

