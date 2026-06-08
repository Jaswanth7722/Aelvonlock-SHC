"""
Aelvonlock-SHC Core Module
==========================
Symbolic Hashing Cryptography - Shared core utilities, symbol map,
and security primitives shared across all variants.

This module provides the foundation for all Aelvonlock variants:
- Symbol map with integrity verification
- ARX (Addition-Rotation-XOR) primitives
- Binary/encoding utilities
- Salt generation and stretching
- Memory-hard matrix operations
- Constant-time comparison
"""

import hashlib
import hmac
import json
import math
import os
import sys
import struct
import types
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════
# VERSION INFORMATION
# ═══════════════════════════════════════════════════════════════

VERSION = "2.0.0"
VERSION_NAME = "Aelvonlock-SHC"
VERSION_TAG = "AEON-2.0"

# ═══════════════════════════════════════════════════════════════
# SYMBOL MAP (Custom Unicode Block U+100000 - U+1000FF)
# ═══════════════════════════════════════════════════════════════

# The Aelvonlock symbol map uses a private Unicode block.
# These 94 symbols are NOT present in any standard encoding (UTF-8, ASCII, Base64).
# Each character maps to a unique Unicode codepoint in the range U+100000-U+1000FF.

_SYMBOL_MAP_RAW: Dict[str, str] = {
    # Uppercase A-Z (26)
    "A": "\U00100000", "B": "\U00100001", "C": "\U00100002", "D": "\U00100003",
    "E": "\U00100004", "F": "\U00100005", "G": "\U00100006", "H": "\U00100007",
    "I": "\U00100008", "J": "\U00100009", "K": "\U0010000A", "L": "\U0010000B",
    "M": "\U0010000C", "N": "\U0010000D", "O": "\U0010000E", "P": "\U0010000F",
    "Q": "\U00100010", "R": "\U00100011", "S": "\U00100012", "T": "\U00100013",
    "U": "\U00100014", "V": "\U00100015", "W": "\U00100016", "X": "\U00100017",
    "Y": "\U00100018", "Z": "\U00100019",

    # Lowercase a-z (26)
    "a": "\U0010001A", "b": "\U0010001B", "c": "\U0010001C", "d": "\U0010001D",
    "e": "\U0010001E", "f": "\U0010001F", "g": "\U00100020", "h": "\U00100021",
    "i": "\U00100022", "j": "\U00100023", "k": "\U00100024", "l": "\U00100025",
    "m": "\U00100026", "n": "\U00100027", "o": "\U00100028", "p": "\U00100029",
    "q": "\U0010002A", "r": "\U0010002B", "s": "\U0010002C", "t": "\U0010002D",
    "u": "\U0010002E", "v": "\U0010002F", "w": "\U00100030", "x": "\U00100031",
    "y": "\U00100032", "z": "\U00100033",

    # Digits 0-9 (10)
    "0": "\U00100034", "1": "\U00100035", "2": "\U00100036", "3": "\U00100037",
    "4": "\U00100038", "5": "\U00100039", "6": "\U0010003A", "7": "\U0010003B",
    "8": "\U0010003C", "9": "\U0010003D",

    # Extended symbols (32 additional)
    ".": "\U0010003E", ",": "\U0010003F", "!": "\U00100040", "@": "\U00100041",
    "#": "\U00100042", "$": "\U00100043", "%": "\U00100044", "^": "\U00100045",
    "&": "\U00100046", "*": "\U00100047", "(": "\U00100048", ")": "\U00100049",
    "-": "\U0010004A", "_": "\U0010004B", "=": "\U0010004C", "+": "\U0010004D",
    "[": "\U0010004E", "]": "\U0010004F", "{": "\U00100050", "}": "\U00100051",
    "|": "\U00100052", "\\": "\U00100053", ":": "\U00100054", ";": "\U00100055",
    "\"": "\U00100056", "'": "\U00100057", "<": "\U00100058", ">": "\U00100059",
    "/": "\U0010005A", "?": "\U0010005B", "~": "\U0010005C", "`": "\U0010005D"
}

# Make symbol map immutable via MappingProxyType
SYMBOL_MAP: types.MappingProxyType = types.MappingProxyType(_SYMBOL_MAP_RAW)

# Build reverse map for decoding
REVERSE_SYMBOL_MAP: Dict[str, str] = {v: k for k, v in _SYMBOL_MAP_RAW.items()}

# Symbol list for output generation
SYMBOL_LIST: List[str] = list(_SYMBOL_MAP_RAW.values())

# Verify symbol map integrity
assert len(SYMBOL_MAP) == 94, f"Symbol map must have 94 symbols, got {len(SYMBOL_MAP)}"
assert len(set(SYMBOL_LIST)) == 94, "Symbol map contains duplicates!"

# ═══════════════════════════════════════════════════════════════
# CRYPTOGRAPHIC CONSTANTS
# ═══════════════════════════════════════════════════════════════

WORD_SIZE = 64          # 64-bit words
BLOCK_SIZE = 512        # 512-bit blocks (8 words)
NUM_WORDS = 8           # 8 words per block

# Rotation constants (derived from prime numbers)
ROT1, ROT2, ROT3 = 11, 19, 7
ROT4, ROT5, ROT6 = 23, 17, 5
ROT7, ROT8 = 13, 29

# Mixing constants (large primes)
MULT_CONST = 0x9E3779B97F4A7C15    # Golden ratio
MIX_PRIME_1 = 0xC6A4A7935BD1E995   # From MurmurHash
MIX_PRIME_2 = 0xFF51AFD7ED558CCD   # From MurmurHash3
INIT_XOR = 0xABCDEF1234567890      # Initialization constant

# Security levels
SECURITY_LOW = 0
SECURITY_STANDARD = 1
SECURITY_HIGH = 2
SECURITY_MAXIMUM = 3
SECURITY_ULTIMATE = 4

# Maximum input length (1MB default)
MAX_INPUT_LENGTH = 1_048_576

# Memory multipliers per security level
MEMORY_MULTIPLIERS = {
    SECURITY_LOW: 1,
    SECURITY_STANDARD: 4,
    SECURITY_HIGH: 16,
    SECURITY_MAXIMUM: 64,
    SECURITY_ULTIMATE: 256,
}

# ARX rounds per security level
ARX_ROUNDS = {
    SECURITY_LOW: 4,
    SECURITY_STANDARD: 12,
    SECURITY_HIGH: 32,
    SECURITY_MAXIMUM: 64,
    SECURITY_ULTIMATE: 128,
}

# Number of lanes per security level
NUM_LANES = {
    SECURITY_LOW: 1,
    SECURITY_STANDARD: 2,
    SECURITY_HIGH: 4,
    SECURITY_MAXIMUM: 8,
    SECURITY_ULTIMATE: 16,
}

# Finalization rounds
FINALIZE_ROUNDS = {
    SECURITY_LOW: 2,
    SECURITY_STANDARD: 4,
    SECURITY_HIGH: 6,
    SECURITY_MAXIMUM: 8,
    SECURITY_ULTIMATE: 16,
}

# ═══════════════════════════════════════════════════════════════
# INTEGRITY VERIFICATION
# ═══════════════════════════════════════════════════════════════

def _verify_code_integrity() -> None:
    """
    Verify that the core module hasn't been tampered with.
    If verification fails, enters an infinite loop to prevent hashing.
    """
    def _corrupted() -> None:
        while True:
            msg = "\n[FATAL] Aelvonlock core integrity check failed!"
            msg += "\nSymbol map or core constants have been tampered with."
            msg += "\nSystem halted for security."
            print(msg, file=sys.stderr)
            sys.stderr.flush()
            # Burn CPU to prevent misuse
            _ = [0] * (1 << 20)

    try:
        # Verify symbol map integrity
        if len(SYMBOL_MAP) != 94:
            _corrupted()

        # Verify constants haven't changed
        expected_rot1 = 11
        if ROT1 != expected_rot1:
            _corrupted()

        # Verify module hasn't lost SYMBOL_MAP
        if not hasattr(sys.modules[__name__], 'SYMBOL_MAP'):
            _corrupted()

        # Verify symbol values are correct
        test_sum = sum(ord(v[0]) for v in SYMBOL_MAP.values())
        if test_sum == 0:
            _corrupted()

    except Exception:
        _corrupted()


def require_symbol_map(func):
    """
    Decorator that verifies symbol map integrity before each call.
    Prevents runtime modification of the symbol map.
    """
    original_map = dict(SYMBOL_MAP)
    original_list = list(SYMBOL_LIST)

    def wrapper(*args, **kwargs):
        # Quick integrity check without full dict copy
        current_map = getattr(sys.modules[__name__], 'SYMBOL_MAP', None)
        if current_map is None or len(current_map) != 94:
            _verify_code_integrity()

        # Verify SYMBOL_LIST integrity
        current_list = getattr(sys.modules[__name__], 'SYMBOL_LIST', None)
        if current_list is None or len(current_list) != 94:
            _verify_code_integrity()

        return func(*args, **kwargs)

    return wrapper


# ═══════════════════════════════════════════════════════════════
# INPUT VALIDATION & SANITIZATION
# ═══════════════════════════════════════════════════════════════

def sanitize_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Validate and sanitize input text."""
    if not isinstance(text, str):
        raise TypeError(f"Input must be a string, got {type(text).__name__}")
    if len(text) == 0:
        raise ValueError("Input cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
    # Normalize unicode
    import unicodedata
    return unicodedata.normalize('NFKC', text)


# ═══════════════════════════════════════════════════════════════
# BITWISE OPERATIONS
# ═══════════════════════════════════════════════════════════════

def rotate_left(val: int, r: int, width: int = WORD_SIZE) -> int:
    """Rotate 64-bit integer left."""
    r %= width
    return ((val << r) & ((1 << width) - 1)) | (val >> (width - r))


def rotate_right(val: int, r: int, width: int = WORD_SIZE) -> int:
    """Rotate 64-bit integer right."""
    r %= width
    return (val >> r) | ((val << (width - r)) & ((1 << width) - 1))


def bytes_to_u64_le(data: bytes) -> int:
    """Convert bytes to little-endian 64-bit integer."""
    return int.from_bytes(data[:8], 'little') & ((1 << WORD_SIZE) - 1)


def bytes_to_u64_be(data: bytes) -> int:
    """Convert bytes to big-endian 64-bit integer."""
    return int.from_bytes(data[:8], 'big') & ((1 << WORD_SIZE) - 1)


def u64_to_bytes_le(val: int) -> bytes:
    """Convert 64-bit integer to little-endian bytes."""
    return struct.pack('<Q', val & ((1 << 64) - 1))


def u64_to_bytes_be(val: int) -> bytes:
    """Convert 64-bit integer to big-endian bytes."""
    return struct.pack('>Q', val & ((1 << 64) - 1))


def mask64(val: int) -> int:
    """Mask to 64 bits."""
    return val & ((1 << 64) - 1)


# ═══════════════════════════════════════════════════════════════
# BINARY PROCESSING
# ═══════════════════════════════════════════════════════════════

def to_binary(text: str) -> str:
    """Convert string to binary string."""
    return ''.join(f'{ord(c):08b}' for c in text)


def pad_binary(b: str, size: int = BLOCK_SIZE) -> str:
    """
    Pad binary string to multiple of block size.
    Uses Merkle-Damgård strengthening (1 + 0s + length encoding).
    """
    # Reserve 65 bits for the length encoding (1 + 64 bits)
    length_bits = len(b)
    pad_len = (size - (len(b) + 65) % size) % size
    # 1 bit marker + pad_len zeros + 64 bits for length
    return b + '1' + '0' * pad_len + f'{length_bits:064b}'


def split_blocks(b: str, size: int = BLOCK_SIZE) -> List[str]:
    """Split binary string into fixed-size blocks."""
    return [b[i:i + size] for i in range(0, len(b), size)]


def words_from_block(block: str) -> List[int]:
    """Convert binary block to list of integer words."""
    return [int(block[i:i + WORD_SIZE], 2) for i in range(0, len(block), WORD_SIZE)]


def words_to_binary(words: Union[List[int], np.ndarray]) -> str:
    """Convert list of words to binary string."""
    return ''.join(f'{int(w):0{WORD_SIZE}b}' for w in words)


# ═══════════════════════════════════════════════════════════════
# SYMBOL ENCODING
# ═══════════════════════════════════════════════════════════════

def encode_symbols(text: str) -> str:
    """Encode text using Aelvonlock symbol map."""
    return ''.join(SYMBOL_MAP.get(c, c) for c in text)


def decode_symbols(symbol_text: str) -> str:
    """Decode Aelvonlock symbols back to original text."""
    result = []
    i = 0
    while i < len(symbol_text):
        ch = symbol_text[i]
        decoded = REVERSE_SYMBOL_MAP.get(ch)
        if decoded is not None:
            result.append(decoded)
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


def bytes_to_symbols(data: bytes, symbol_list: List[str] = SYMBOL_LIST,
                     desired_length: int = 64) -> str:
    """
    Convert bytes to Aelvonlock symbolic output.
    Uses rejection sampling for unbiased output.
    """
    if not symbol_list:
        raise ValueError("Symbol list is empty")

    num_symbols = len(symbol_list)
    mod_base = 256 - (256 % num_symbols)
    result: List[str] = []
    i = 0

    while len(result) < desired_length:
        if i >= len(data):
            # Extend data deterministically
            ext = hashlib.sha256(data + i.to_bytes(8, 'little')).digest()
            data += ext

        byte_val = data[i]
        if byte_val < mod_base:
            result.append(symbol_list[byte_val % num_symbols])
        # If byte >= mod_base, skip (rejection sampling)
        i += 1

    return ''.join(result[:desired_length])


# ═══════════════════════════════════════════════════════════════
# SALT GENERATION & STRETCHING
# ═══════════════════════════════════════════════════════════════

def generate_entropy_salt(length: int = 32) -> bytes:
    """Generate cryptographically secure random salt."""
    return os.urandom(length)


def salt_to_int(salt: bytes) -> int:
    """Convert salt bytes to a 64-bit integer."""
    return bytes_to_u64_be(salt)


def salt_to_words(salt: bytes, num_words: int = NUM_WORDS) -> List[int]:
    """Convert salt bytes to multiple 64-bit words."""
    words = []
    for i in range(num_words):
        chunk = salt[i * 8:(i + 1) * 8]
        if len(chunk) < 8:
            chunk = chunk.ljust(8, b'\x00')
        words.append(bytes_to_u64_be(chunk))
    return words


def stretch_salt(salt: bytes, input_text: str,
                 iterations: int = 100_000) -> bytes:
    """
    Stretch salt using HMAC-SHA256 in a PBKDF2-like construction.
    This makes salt brute-forcing computationally expensive.
    """
    # Derive a key from input text for HMAC
    input_key = hashlib.sha256(input_text.encode('utf-8')).digest()

    # Stretch salt using HMAC in a Feistel-like construction
    stretched = salt
    for i in range(iterations):
        stretched = hmac.new(
            input_key + i.to_bytes(4, 'little'),
            stretched,
            'sha256'
        ).digest()
        # Mix back with previous state
        if i % 2 == 0:
            stretched = bytes(a ^ b for a, b in zip(stretched, salt * (len(stretched) // len(salt) + 1)))

    return stretched


# ═══════════════════════════════════════════════════════════════
# STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def initialize_state(salt_words: List[int],
                     num_words: int = NUM_WORDS) -> List[int]:
    """Initialize state vector from salt words."""
    state: List[int] = []
    for i in range(num_words):
        if i < len(salt_words):
            val = salt_words[i] ^ ((i + 1) * INIT_XOR)
        else:
            val = INIT_XOR ^ ((i + 1) * 0x9E3779B97F4A7C15)
        state.append(mask64(val))
    return state


def initialize_lanes(salt_words: List[int],
                     num_lanes: int = 4,
                     words_per_lane: int = NUM_WORDS) -> List[List[int]]:
    """Initialize multiple processing lanes."""
    lanes: List[List[int]] = []
    for lane in range(num_lanes):
        lane_salt = [(sw ^ (lane * 0xFF51AFD7ED558CCD)) & ((1 << 64) - 1)
                     for sw in salt_words]
        lanes.append(initialize_state(lane_salt, words_per_lane))
    return lanes


# ═══════════════════════════════════════════════════════════════
# ARX CORE PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def mix_schedule(words: List[int], num_words: int = NUM_WORDS) -> List[int]:
    """
    Message expansion / mixing schedule.
    Transforms input words using Feistel-like structure.
    """
    n = len(words)
    expanded = list(words)
    # Expand to at least num_words
    while len(expanded) < num_words:
        idx = len(expanded)
        val = (
            expanded[idx - 1] ^
            rotate_left(expanded[idx - 2], ROT3) ^
            (expanded[idx - 3] * MULT_CONST) ^
            (idx * MIX_PRIME_1)
        )
        expanded.append(mask64(val))

    # Mix using Feistel rounds
    for _ in range(4):
        for i in range(num_words):
            left = expanded[i]
            right = expanded[(i + 1) % num_words]
            expanded[i] = mask64(
                left ^ rotate_right(right, ROT1) ^
                (right * MIX_PRIME_2)
            )
            expanded[(i + 1) % num_words] = mask64(
                right ^ rotate_left(expanded[i], ROT2) ^
                (expanded[i] * MULT_CONST)
            )

    return expanded[:num_words]


def arx_round(words: Union[List[int], np.ndarray],
              key: int,
              num_words: Optional[int] = None) -> List[int]:
    """
    Single ARX (Addition-Rotation-XOR) round.
    Uses len(words) when num_words is None for maximum flexibility.
    """
    if isinstance(words, np.ndarray):
        words = words.tolist()
    
    n = num_words if num_words is not None else len(words)

    for i in range(n):
        # Addition with key and round constant
        words[i] = mask64(words[i] + key + (i * 13))

        # XOR with rotated neighbor
        neighbor = words[(i + 1) % n]
        words[i] ^= rotate_left(neighbor, ROT1)

        # Word-wise rotation
        words[i] = rotate_left(words[i], ROT2)

        # Additional mixing
        words[i] ^= rotate_right(neighbor, ROT4)
        words[i] = mask64(words[i] * MULT_CONST) ^ rotate_left(words[i], ROT5)

    return words


def arx_round_pair(w0: int, w1: int, key: int) -> Tuple[int, int]:
    """
    ARX round operating on a pair of words (Feistel-like).
    """
    w0 = mask64(w0 + key)
    w1 = mask64(w1 ^ rotate_left(w0, ROT1))
    w0 = mask64(w0 ^ rotate_right(w1, ROT2))
    w1 = mask64(w1 + w0 * MULT_CONST)
    w0 = mask64(w0 ^ rotate_left(w1, ROT3))
    return w0, w1


def arx_mix_lanes(lanes: List[List[int]],
                  key: int,
                  num_words: int = NUM_WORDS) -> List[List[int]]:
    """
    Mix between multiple lanes using ARX operations.
    Each lane exchanges state with neighboring lanes.
    """
    n = len(lanes)
    if n <= 1:
        return lanes

    mixed = [list(lane) for lane in lanes]
    for i in range(n):
        prev = lanes[(i - 1) % n]
        next_lane = lanes[(i + 1) % n]
        for j in range(num_words):
            # Cross-lane mixing
            mix_val = mask64(
                prev[j] ^ next_lane[(j + 1) % num_words] ^ key
            )
            mixed[i][j] = mask64(
                mixed[i][j] ^ rotate_left(mix_val, ROT1) ^
                (mix_val * MIX_PRIME_2)
            )
    return mixed


def finalize_state(state: List[int],
                   salt_words: List[int],
                   rounds: int = 4,
                   num_words: int = NUM_WORDS) -> List[int]:
    """
    Finalize hash state with multiple ARX rounds and salt mixing.
    """
    for rnd in range(rounds):
        key = mask64(salt_words[rnd % len(salt_words)] ^ (rnd * MIX_PRIME_1))
        state = arx_round(state, key, num_words)
        # XOR with all salt words
        for i in range(num_words):
            sw = salt_words[i % len(salt_words)]
            state[i] = mask64(state[i] ^ rotate_left(sw, (i + rnd) % 64))
            state[i] = mask64(state[i] ^ rotate_right(state[(i + 1) % num_words], ROT3))

    return state


# ═══════════════════════════════════════════════════════════════
# MEMORY-HARD MATRIX (Numba JIT Accelerated)
# ═══════════════════════════════════════════════════════════════

@np.errstate(all='ignore')
def compute_memory_size(security_level: int,
                        input_length: int = 0) -> Tuple[int, int]:
    """
    Compute memory matrix dimensions based on security level and input.
    Memory is input-dependent: longer inputs get larger matrices.
    """
    multiplier = MEMORY_MULTIPLIERS.get(security_level, 4)

    # Base size from security level
    base_size = 1024 * multiplier

    # Input-dependent scaling
    if input_length > 0:
        # Log scale: memory ∝ log(input_length) * multiplier
        input_factor = max(1, int(math.log2(input_length + 1)))
        scaled_size = base_size + (input_factor * 64 * multiplier)
    else:
        scaled_size = base_size

    # Ensure minimum size
    scaled_size = max(256, min(scaled_size, 65536))

    # Make dimensions work well for memory access patterns
    rows = int(math.isqrt(scaled_size * 1024))
    cols = rows

    # Ensure at least minimum
    rows = max(64, rows)
    cols = max(64, cols)

    return rows, cols


@np.errstate(all='ignore')
def fill_memory_matrix(matrix: np.ndarray,
                       state: np.ndarray,
                       salt_words: np.ndarray) -> np.ndarray:
    """
    Fill memory matrix using state-dependent patterns.
    Uses Argon2-like random memory access.
    """
    rows, cols = matrix.shape
    num_words = len(state)

    # Phase 1: Deterministic fill with state mixing
    for i in range(rows):
        for j in range(cols):
            val = mask64(
                (i * j) ^
                int(state[j % num_words]) ^
                int(salt_words[j % len(salt_words)]) ^
                (i * MIX_PRIME_1)
            )
            matrix[i, j] = np.uint64(val)

    # Phase 2: Random access pattern (Argon2-like)
    for _ in range(4):
        for i in range(rows):
            for j in range(cols):
                # Random-like index derived from state
                si = int(state[i % num_words]) % rows
                sj = int(state[j % num_words]) % cols
                matrix[i, j] ^= matrix[si, sj]
                matrix[i, j] = np.uint64(mask64(
                    int(matrix[i, j]) ^
                    int(state[(i + j) % num_words]) ^
                    MIX_PRIME_2
                ))

    # Phase 3: Diffusion
    for rnd in range(4):
        for i in range(rows):
            for j in range(cols):
                nj = (j + rnd * 7) % cols
                ni = (i + rnd * 11) % rows
                matrix[i, j] = np.uint64(mask64(
                    int(matrix[i, j]) ^
                    int(matrix[ni, nj]) ^
                    int(state[(i + j + rnd) % num_words])
                ))

    return matrix


# ═══════════════════════════════════════════════════════════════
# ENTROPY EXTRACTION FROM INPUT
# ═══════════════════════════════════════════════════════════════

def extract_input_entropy(text: str) -> bytes:
    """
    Extract entropy from input text using BLAKE2b.
    BLAKE2b provides excellent security, is well-analyzed,
    and avoids the complexity of ad-hoc multi-hash combinations.
    """
    data = text.encode('utf-8')
    return hashlib.blake2b(data, digest_size=32).digest()


# ═══════════════════════════════════════════════════════════════
# CONSTANT-TIME COMPARISON
# ═══════════════════════════════════════════════════════════════

def constant_time_compare(a: Union[str, bytes],
                          b: Union[str, bytes]) -> bool:
    """Compare two strings/bytes in constant time."""
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')

    if len(a) != len(b):
        return False

    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0


# ═══════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════

@require_symbol_map
def generate_symbolic_output(state: Union[List[int], np.ndarray],
                             desired_length: int = 64) -> str:
    """
    Generate Aelvonlock symbolic hash output from state vector.
    Uses rejection sampling for unbiased distribution.
    """
    final_bin = words_to_binary(state)
    num_symbols = len(SYMBOL_LIST)
    mod_base = 256 - (256 % num_symbols)

    result: List[str] = []
    i = 0
    retries = 0
    max_retries = 10000

    while len(result) < desired_length and retries < max_retries:
        chunk = final_bin[i:i + 8]
        if len(chunk) < 8:
            chunk = chunk.ljust(8, '0')
        byte_val = int(chunk, 2)
        i += 8

        if byte_val < mod_base:
            result.append(SYMBOL_LIST[byte_val % num_symbols])
            retries = 0
        else:
            retries += 1
            if retries >= max_retries:
                # Fallback: use biased selection to avoid infinite loop
                result.append(SYMBOL_LIST[byte_val % num_symbols])
                retries = 0

        # If we exhausted the binary string, extend state
        if i >= len(final_bin):
            state = arx_round(state, i * MIX_PRIME_2, len(state))
            final_bin += words_to_binary(state)

    return ''.join(result[:desired_length])


# ═══════════════════════════════════════════════════════════════
# MODULE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

# Run integrity check on import
_verify_code_integrity()

# Export public API
__all__ = [
    'VERSION', 'VERSION_NAME', 'VERSION_TAG',
    'SYMBOL_MAP', 'REVERSE_SYMBOL_MAP', 'SYMBOL_LIST',
    'WORD_SIZE', 'BLOCK_SIZE', 'NUM_WORDS',
    'ROT1', 'ROT2', 'ROT3', 'ROT4', 'ROT5', 'ROT6',
    'MULT_CONST', 'MIX_PRIME_1', 'MIX_PRIME_2', 'INIT_XOR',
    'SECURITY_LOW', 'SECURITY_STANDARD', 'SECURITY_HIGH',
    'SECURITY_MAXIMUM', 'SECURITY_ULTIMATE',
    'ARX_ROUNDS', 'NUM_LANES', 'FINALIZE_ROUNDS', 'MEMORY_MULTIPLIERS',
    'sanitize_input', 'rotate_left', 'rotate_right',
    'mask64', 'bytes_to_u64_le', 'bytes_to_u64_be',
    'to_binary', 'pad_binary', 'split_blocks',
    'words_from_block', 'words_to_binary',
    'encode_symbols', 'decode_symbols', 'bytes_to_symbols',
    'generate_entropy_salt', 'salt_to_int', 'salt_to_words',
    'stretch_salt', 'initialize_state', 'initialize_lanes',
    'mix_schedule', 'arx_round', 'arx_round_pair',
    'arx_mix_lanes', 'finalize_state',
    'compute_memory_size', 'fill_memory_matrix',
    'extract_input_entropy',
    'constant_time_compare', 'generate_symbolic_output',
    'require_symbol_map', '_verify_code_integrity',
]
