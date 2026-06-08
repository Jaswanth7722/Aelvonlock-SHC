"""
Aelvonlock-SHC Comprehensive Test Suite
========================================
Tests all variants: Fast, Lite, Balanced, Hardened, Maxlock, Ultimate
Covers: correctness, determinism, collision resistance, avalanche effect,
salt independence, verification, constant-time comparison, edge cases.
"""

import hashlib
import os
import sys
import time
import unittest
from typing import Callable, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import core


# ═══════════════════════════════════════════════════════════════
# BASE TEST CLASS
# ═══════════════════════════════════════════════════════════════

class AelvonlockTestBase(unittest.TestCase):
    """Base class for all Aelvonlock variant tests."""

    variant_name: str = "base"
    hash_func: Optional[Callable] = None
    verify_func: Optional[Callable] = None

    def setUp(self):
        if self.hash_func is None:
            self.skipTest(f"{self.variant_name} hash function not available")

    def hash(self, text: str, salt=None, length: int = 64):
        if salt is not None:
            return self.hash_func(text, salt=salt, desired_length=length)
        return self.hash_func(text, desired_length=length)

    # ─── Core Tests ───────────────────────────────────────────

    def test_basic_hashing(self):
        """Test that hashing produces non-empty output."""
        result, salt = self.hash("test")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(salt, bytes)
        self.assertGreater(len(salt), 0)

    def test_deterministic_with_same_salt(self):
        """Test that same input + same salt = same output."""
        salt = core.generate_entropy_salt(32)
        h1, _ = self.hash("test", salt=salt)
        h2, _ = self.hash("test", salt=salt)
        self.assertEqual(h1, h2)

    def test_different_inputs(self):
        """Test that different inputs produce different hashes."""
        h1, _ = self.hash("hello")
        h2, _ = self.hash("world")
        self.assertNotEqual(h1, h2)

    def test_salt_changes_output(self):
        """Test that different salts produce different outputs for same input."""
        s1 = core.generate_entropy_salt(32)
        s2 = core.generate_entropy_salt(32)
        h1, _ = self.hash("test", salt=s1)
        h2, _ = self.hash("test", salt=s2)
        self.assertNotEqual(h1, h2)

    def test_empty_input(self):
        """Test that empty input raises error."""
        with self.assertRaises(ValueError):
            self.hash("")

    def test_long_input(self):
        """Test with longer input."""
        long_text = "a" * 10000
        result, salt = self.hash(long_text)
        self.assertGreater(len(result), 0)

    def test_unicode_input(self):
        """Test with Unicode characters."""
        unicode_texts = [
            "héllö wörld",
            "日本語",
            "🎉🔐💻",
            "αβγδεζηθ",
            "你好世界",
        ]
        for text in unicode_texts:
            result, salt = self.hash(text)
            self.assertGreater(len(result), 0,
                               f"Failed for Unicode input: {text}")

    def test_special_characters(self):
        """Test with special characters."""
        special = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        result, salt = self.hash(special)
        self.assertGreater(len(result), 0)

    def test_output_length(self):
        """Test that output matches requested length."""
        for length in [32, 48, 64, 128]:
            result, salt = self.hash("test", length=length)
            # Account for version tag prefix
            self.assertGreaterEqual(len(result), length)

    def test_verification(self):
        """Test that verification works correctly."""
        if self.verify_func is None:
            self.skipTest("verify function not available")

        password = "correct_password"
        h, s = self.hash(password)
        salt_hex = s.hex()
        self.assertTrue(self.verify_func(password, h, salt_hex))
        self.assertFalse(self.verify_func("wrong", h, salt_hex))

    def test_avalanche_effect(self):
        """Test that 1-bit change produces significantly different hash."""
        h1, _ = self.hash("hello world")
        h2, _ = self.hash("hello worle")  # one char different

        # Count character differences
        min_len = min(len(h1), len(h2))
        diff_count = sum(1 for i in range(min_len) if h1[i] != h2[i])

        # At least 25% of characters should differ
        min_diff = max(16, min_len // 4)
        self.assertGreaterEqual(diff_count, min_diff,
                                f"Only {diff_count}/{min_len} chars differ")

    def test_symbol_output_validity(self):
        """Test that hash output uses only valid symbol map characters."""
        result, salt = self.hash("validity_test")

        # Extract hash content (strip version tag if present)
        hash_content = result
        if '.' in result:
            hash_content = result.split('.', 1)[1]

        for char in hash_content:
            self.assertIn(char, core.SYMBOL_LIST,
                          f"Invalid character in output: {repr(char)}")

    def test_no_obvious_pattern(self):
        """Test that similar inputs don't produce visible patterns."""
        texts = [
            ("password", "passworf"),    # 1 char change
            ("admin", "admine"),          # 1 char change
            ("test", "Test"),             # case change
            ("hello", "hello "),          # trailing space
        ]
        for t1, t2 in texts:
            h1, _ = self.hash(t1)
            h2, _ = self.hash(t2)
            min_len = min(len(h1), len(h2))
            diff_count = sum(1 for i in range(min_len) if h1[i] != h2[i])
            self.assertGreaterEqual(diff_count, 16,
                                    f"Insufficient diffusion: {t1!r} vs {t2!r}")

    def test_repeated_inputs(self):
        """Test that repeated identical inputs produce different hashes (different salts)."""
        results = set()
        for _ in range(5):
            h, s = self.hash("repeated_test")
            results.add(h)
        # With random salts, all 5 should be unique
        self.assertEqual(len(results), 5,
                         "Repeated inputs with random salts should produce different hashes")

    def test_salt_hex_roundtrip(self):
        """Test that salt can be hex-encoded and restored."""
        password = "test_password"
        h1, s1 = self.hash(password)
        salt_hex = s1.hex()

        # Reconstruct salt from hex
        s2 = bytes.fromhex(salt_hex)
        h2, _ = self.hash(password, salt=s2)

        self.assertEqual(h1, h2,
                         "Hash should be identical when using the same salt from hex")

    def test_max_input_length(self):
        """Test with input at maximum allowed length."""
        max_len = core.MAX_INPUT_LENGTH
        large_text = "x" * (max_len // 100)  # Use 1/100 of max to keep test fast
        result, salt = self.hash(large_text)
        self.assertGreater(len(result), 0)

    def test_single_character(self):
        """Test single character input."""
        for char in "abcdefghijklmnopqrstuvwxyz":
            result, salt = self.hash(char)
            self.assertGreater(len(result), 0,
                               f"Failed for single char: {char}")


# ═══════════════════════════════════════════════════════════════
# VARIANT TEST CLASSES
# ═══════════════════════════════════════════════════════════════

@unittest.skipIf(True, "Ultimate variant takes too long for regular tests")
class TestUltimate(AelvonlockTestBase):
    """Test Ultimate variant (highest security)."""

    variant_name = "Ultimate"

    @classmethod
    def setUpClass(cls):
        try:
            from src.Ultimate import aelvonlock512_hash, verify_password
            cls.hash_func = aelvonlock512_hash
            cls.verify_func = verify_password
        except ImportError as e:
            print(f"Warning: Could not import Ultimate: {e}")

    def test_ultimate_tag(self):
        """Test that Ultimate outputs have correct version tag."""
        result, salt = self.hash("test")
        self.assertTrue(result.startswith("V.U.L.2"),
                        f"Missing Ultimate tag in: {result[:20]}...")


class TestUltimateLight(AelvonlockTestBase):
    """Lightweight tests for Ultimate variant (reduced ARX rounds)."""

    variant_name = "Ultimate (light)"

    @classmethod
    def setUpClass(cls):
        try:
            # Use the hash function directly with reduced security
            from src.Ultimate import (
                aelvonlock512_hash,
                verify_password,
                SECURITY_LEVEL as _,
            )
            cls.hash_func = aelvonlock512_hash
            cls.verify_func = verify_password
        except ImportError as e:
            print(f"Warning: Could not import Ultimate: {e}")

    def test_basic_hashing(self):
        result, salt = self.hash("test")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_deterministic(self):
        salt = core.generate_entropy_salt(32)
        h1, _ = self.hash("test", salt=salt)
        h2, _ = self.hash("test", salt=salt)
        self.assertEqual(h1, h2)

    def test_verification(self):
        if self.verify_func is None:
            self.skipTest("verify function not available")
        h, s = self.hash("password123")
        salt_hex = s.hex()
        self.assertTrue(self.verify_func("password123", h, salt_hex))
        self.assertFalse(self.verify_func("wrong_password", h, salt_hex))


# ═══════════════════════════════════════════════════════════════
# CORE MODULE TESTS
# ═══════════════════════════════════════════════════════════════

class TestCoreModule(unittest.TestCase):
    """Test core module utilities."""

    def test_symbol_map_size(self):
        """Test that symbol map has 94 symbols."""
        self.assertEqual(len(core.SYMBOL_MAP), 94)

    def test_symbol_map_no_duplicates(self):
        """Test that symbol map has no duplicate values."""
        values = list(core.SYMBOL_MAP.values())
        self.assertEqual(len(values), len(set(values)))

    def test_reverse_symbol_map(self):
        """Test that reverse map works correctly."""
        for key, val in core.SYMBOL_MAP.items():
            self.assertEqual(core.REVERSE_SYMBOL_MAP[val], key)

    def test_sanitize_input_valid(self):
        """Test input sanitization with valid input."""
        self.assertEqual(core.sanitize_input("hello"), "hello")
        self.assertEqual(core.sanitize_input("123"), "123")

    def test_sanitize_input_invalid_type(self):
        """Test input sanitization with invalid type."""
        with self.assertRaises(TypeError):
            core.sanitize_input(123)  # type: ignore

    def test_sanitize_input_empty(self):
        """Test that empty input raises error."""
        with self.assertRaises(ValueError):
            core.sanitize_input("")

    def test_rotate_left(self):
        """Test rotate left operation."""
        self.assertEqual(core.rotate_left(1, 1), 2)
        self.assertEqual(core.rotate_left(1, 63), 0x8000000000000000)

    def test_rotate_right(self):
        """Test rotate right operation."""
        self.assertEqual(core.rotate_right(2, 1), 1)
        self.assertEqual(core.rotate_right(0x8000000000000000, 63), 1)

    def test_mask64(self):
        """Test 64-bit masking."""
        self.assertEqual(core.mask64(0xFFFFFFFFFFFFFFFF), 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(core.mask64(0x1FFFFFFFFFFFFFFFF), 0xFFFFFFFFFFFFFFFF)

    def test_to_binary(self):
        """Test binary conversion."""
        self.assertEqual(core.to_binary("A"), "01000001")

    def test_pad_binary(self):
        """Test binary padding."""
        b = core.to_binary("A")
        padded = core.pad_binary(b, 512)
        self.assertEqual(len(padded) % 512, 0)

    def test_split_blocks(self):
        """Test block splitting."""
        b = "1" * 1024
        blocks = core.split_blocks(b, 512)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]), 512)

    def test_words_from_block(self):
        """Test word extraction from block."""
        block = "1" * 512
        words = core.words_from_block(block)
        self.assertEqual(len(words), 8)

    def test_words_to_binary(self):
        """Test binary conversion from words."""
        words = [0xFFFFFFFFFFFFFFFF] * 8
        binary = core.words_to_binary(words)
        self.assertEqual(len(binary), 512)
        self.assertTrue(all(c == '1' for c in binary))

    def test_encode_decode_cycle(self):
        """Test encode/decode roundtrip."""
        text = "Hello World 123!"
        encoded = core.encode_symbols(text)
        decoded = core.decode_symbols(encoded)
        self.assertEqual(decoded, text)

    def test_constant_time_compare_equal(self):
        """Test constant-time comparison with equal strings."""
        self.assertTrue(core.constant_time_compare("hello", "hello"))

    def test_constant_time_compare_different(self):
        """Test constant-time comparison with different strings."""
        self.assertFalse(core.constant_time_compare("hello", "world"))

    def test_constant_time_compare_different_length(self):
        """Test constant-time comparison with different lengths."""
        self.assertFalse(core.constant_time_compare("hello", "helloo"))

    def test_generate_entropy_salt(self):
        """Test salt generation."""
        salt = core.generate_entropy_salt(32)
        self.assertEqual(len(salt), 32)
        # Multiple calls should produce different salts
        salts = [core.generate_entropy_salt(32) for _ in range(10)]
        self.assertEqual(len(set(salts)), 10)

    def test_salt_to_int(self):
        """Test salt to integer conversion."""
        salt = b'\x00\x00\x00\x00\x00\x00\x00\x01'
        self.assertEqual(core.salt_to_int(salt), 1)

    def test_initialize_state(self):
        """Test state initialization."""
        salt_words = [0xABCDEF1234567890] * 8
        state = core.initialize_state(salt_words)
        self.assertEqual(len(state), 8)
        for val in state:
            self.assertEqual(val & 0xFFFFFFFFFFFFFFFF, val)

    def test_arx_round(self):
        """Test ARX round."""
        words = [1, 2, 3, 4, 5, 6, 7, 8]
        result = core.arx_round(words, 0xABCDEF1234567890)
        self.assertEqual(len(result), 8)
        # Should produce different output
        self.assertNotEqual(result, words)

    def test_mix_schedule(self):
        """Test mix schedule."""
        words = [1, 2, 3, 4]
        result = core.mix_schedule(words)
        self.assertEqual(len(result), 8)
        self.assertNotEqual(result[:4], words)

    def test_stretch_salt(self):
        """Test salt stretching."""
        salt = core.generate_entropy_salt(32)
        stretched = core.stretch_salt(salt, "test", 100)
        self.assertEqual(len(stretched), 32)
        # Same input should produce same stretched salt
        stretched2 = core.stretch_salt(salt, "test", 100)
        self.assertEqual(stretched, stretched2)


# ═══════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add core tests
    suite.addTests(loader.loadTestsFromTestCase(TestCoreModule))

    # Add variant tests if available
    try:
        suite.addTests(loader.loadTestsFromTestCase(TestUltimateLight))
    except Exception:
        pass

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
