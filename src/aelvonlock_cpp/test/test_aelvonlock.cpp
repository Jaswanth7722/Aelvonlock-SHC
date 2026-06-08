/**
 * @file test_aelvonlock.cpp
 * @brief Test suite for Aelvonlock-SHC C++ library
 */
#include "aelvonlock.hpp"
#include <cstdio>
#include <cstring>
#include <string>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name, expr) do { \
    tests_run++; \
    if (expr) { \
        tests_passed++; \
        printf("  [PASS] %s\n", name); \
    } else { \
        printf("  [FAIL] %s\n", name); \
    } \
} while(0)

int main() {
    printf("Aelvonlock-SHC C++ Library Test Suite\n");
    printf("======================================\n\n");

    /* Test 1: Version */
    TEST("version", aelvonlock::VERSION_STRING == "2.0.0");

    /* Test 2: Symbol map */
    TEST("symbol_count", aelvonlock::NUM_SYMBOLS == 94);
    TEST("first_symbol", aelvonlock::SYMBOLS[0] == 0x100000);
    TEST("last_symbol", aelvonlock::SYMBOLS[93] == 0x10005D);

    /* Test 3: Basic hashing */
    {
        auto [hash, salt] = aelvonlock::hash("test");
        TEST("basic_hashing", !hash.empty());
        TEST("hash_has_tag", hash.substr(0, 8) == "V.C.P.2.");
        TEST("salt_generated", salt.size() == 32);
        printf("           -> hash size: %zu, salt size: %zu\n", hash.size(), salt.size());
    }

    /* Test 4: Deterministic with same salt */
    {
        std::array<uint8_t, 32> fixed_salt = {1, 2, 3, 4};
        auto [h1, _] = aelvonlock::hash("test", fixed_salt);
        auto [h2, __] = aelvonlock::hash("test", fixed_salt);
        TEST("deterministic", h1 == h2);
    }

    /* Test 5: Different inputs */
    {
        auto [h1, _] = aelvonlock::hash("hello");
        auto [h2, __] = aelvonlock::hash("world");
        TEST("different_inputs", h1 != h2);
    }

    /* Test 6: Salt changes output */
    {
        std::array<uint8_t, 32> s1 = {1, 2, 3, 4};
        std::array<uint8_t, 32> s2 = {5, 6, 7, 8};
        auto [h1, _] = aelvonlock::hash("test", s1);
        auto [h2, __] = aelvonlock::hash("test", s2);
        TEST("salt_independence", h1 != h2);
    }

    /* Test 7: Verification */
    {
        auto [hash, salt] = aelvonlock::hash("password123");
        bool valid = aelvonlock::verify("password123", hash, salt);
        bool invalid = aelvonlock::verify("wrong_password", hash, salt);
        TEST("verification_correct", valid);
        TEST("verification_wrong", !invalid);
    }

    /* Test 8: Constant-time compare */
    {
        TEST("constant_time_equal",
             aelvonlock::constant_time_compare("hello", "hello"));
        TEST("constant_time_different",
             !aelvonlock::constant_time_compare("hello", "world"));
        TEST("constant_time_diff_len",
             !aelvonlock::constant_time_compare("hello", "helloo"));
    }

    /* Test 9: Security levels */
    {
        aelvonlock::HashParams params;
        params.security_level = aelvonlock::SecurityLevel::Low;
        params.desired_length = 32;
        aelvonlock::Hasher hasher(params);
        auto [hash, salt] = hasher.hash("test");
        TEST("low_security", !hash.empty());
        printf("           -> Low level hash size: %zu\n", hash.size());
    }

    /* Test 10: Empty input */
    {
        bool caught = false;
        try {
            aelvonlock::hash("");
        } catch (const aelvonlock::EmptyInputError&) {
            caught = true;
        }
        TEST("empty_input_rejected", caught);
    }

    /* Test 11: Output length */
    {
        aelvonlock::HashParams params;
        params.desired_length = 128;
        aelvonlock::Hasher hasher(params);
        auto [hash, salt] = hasher.hash("test");
        TEST("output_length_128", hash.size() >= 128);
        printf("           -> 128-char output: %zu bytes\n", hash.size());
    }

    /* Print summary */
    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);
    printf("Overall: %s\n", tests_passed == tests_run ? "ALL PASSED" : "SOME FAILED");

    return tests_passed == tests_run ? 0 : 1;
}
