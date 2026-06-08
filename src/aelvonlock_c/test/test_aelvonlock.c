/**
 * @file test_aelvonlock.c
 * @brief Test suite for Aelvonlock-SHC C library
 */
#include "aelvonlock.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

int main(void) {
    printf("Aelvonlock-SHC C Library Test Suite\n");
    printf("====================================\n\n");

    /* Initialize */
    int ret = aelvonlock_init();
    TEST("init", ret == AELVONLOCK_SUCCESS);

    /* Version */
    const char* ver = aelvonlock_version();
    TEST("version", strcmp(ver, AELVONLOCK_VERSION_STRING) == 0);

    /* Test 1: Basic hashing */
    {
        const uint8_t input[] = "test";
        uint8_t output[4096];
        size_t out_len = sizeof(output);
        ret = aelvonlock_hash(input, 4, NULL, 0, output, &out_len, NULL);
        TEST("basic_hashing", ret == AELVONLOCK_SUCCESS && out_len > 0);
        if (ret == AELVONLOCK_SUCCESS) {
            printf("           -> %zu bytes output\n", out_len);
        }
    }

    /* Test 2: Deterministic with same salt */
    {
        uint8_t salt[32] = {1, 2, 3};
        uint8_t out1[4096], out2[4096];
        size_t len1 = sizeof(out1), len2 = sizeof(out2);
        aelvonlock_hash((const uint8_t*)"test", 4, salt, 32, out1, &len1, NULL);
        aelvonlock_hash((const uint8_t*)"test", 4, salt, 32, out2, &len2, NULL);
        TEST("deterministic", len1 == len2 && memcmp(out1, out2, len1) == 0);
    }

    /* Test 3: Different inputs */
    {
        uint8_t out1[4096], out2[4096];
        size_t len1 = sizeof(out1), len2 = sizeof(out2);
        int r1 = aelvonlock_hash((const uint8_t*)"hello", 5, NULL, 0, out1, &len1, NULL);
        int r2 = aelvonlock_hash((const uint8_t*)"world", 5, NULL, 0, out2, &len2, NULL);
        TEST("different_inputs",
             r1 == AELVONLOCK_SUCCESS && r2 == AELVONLOCK_SUCCESS &&
             !(len1 == len2 && memcmp(out1, out2, len1) == 0));
    }

    /* Test 4: Salt changes output */
    {
        uint8_t salt1[32] = {1, 2, 3};
        uint8_t salt2[32] = {4, 5, 6};
        uint8_t out1[4096], out2[4096];
        size_t len1 = sizeof(out1), len2 = sizeof(out2);
        aelvonlock_hash((const uint8_t*)"test", 4, salt1, 32, out1, &len1, NULL);
        aelvonlock_hash((const uint8_t*)"test", 4, salt2, 32, out2, &len2, NULL);
        TEST("salt_independence",
             !(len1 == len2 && memcmp(out1, out2, len1) == 0));
    }

    /* Test 5: Verification */
    {
        const uint8_t pw[] = "password123";
        const uint8_t wrong[] = "wrong";
        uint8_t hash[4096];
        size_t hash_len = sizeof(hash);
        uint8_t salt[32] = {0xAB, 0xCD};

        aelvonlock_hash(pw, 11, salt, 32, hash, &hash_len, NULL);
        int valid = aelvonlock_verify(pw, 11, hash, hash_len, salt, 32, NULL);
        int invalid = aelvonlock_verify(wrong, 5, hash, hash_len, salt, 32, NULL);
        TEST("verification_correct", valid == 1);
        TEST("verification_wrong", invalid == 0);
    }

    /* Test 6: Constant-time compare */
    {
        const uint8_t a[] = "hello";
        const uint8_t b[] = "hello";
        const uint8_t c[] = "world";
        TEST("constant_time_equal",
             aelvonlock_constant_time_compare(a, 5, b, 5) == 1);
        TEST("constant_time_different",
             aelvonlock_constant_time_compare(a, 5, c, 5) == 0);
        TEST("constant_time_diff_len",
             aelvonlock_constant_time_compare(a, 5, c, 3) == 0);
    }

    /* Test 7: Empty input rejection */
    {
        uint8_t output[1024];
        size_t out_len = sizeof(output);
        ret = aelvonlock_hash((const uint8_t*)"", 0, NULL, 0, output, &out_len, NULL);
        TEST("empty_input_rejected", ret == AELVONLOCK_ERR_INPUT_EMPTY);
    }

    /* Test 8: Symbol map */
    {
        uint32_t symbols[94];
        int count = aelvonlock_get_symbol_map(symbols);
        TEST("symbol_count", count == 94);
        TEST("first_symbol", symbols[0] == 0x100000);
        TEST("last_symbol", symbols[93] == 0x10005D);
    }

    /* Test 9: Null input handling */
    {
        size_t out_len = 1024;
        uint8_t output[1024];
        ret = aelvonlock_hash(NULL, 0, NULL, 0, output, &out_len, NULL);
        TEST("null_input_handled", ret == AELVONLOCK_ERR_NULL_INPUT);
    }

    /* Test 10: Different output lengths */
    {
        uint8_t output[4096];
        size_t len = 32;
        int r1 = aelvonlock_hash((const uint8_t*)"test", 4, NULL, 0, output, &len, NULL);
        int short_ok = (r1 == AELVONLOCK_SUCCESS && len > 0);

        len = 128;
        int r2 = aelvonlock_hash((const uint8_t*)"test", 4, NULL, 0, output, &len, NULL);
        int long_ok = (r2 == AELVONLOCK_SUCCESS && len > 0);

        TEST("output_lengths", short_ok && long_ok);
    }

    /* Print summary */
    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);
    printf("Overall: %s\n", tests_passed == tests_run ? "ALL PASSED" : "SOME FAILED");

    return tests_passed == tests_run ? 0 : 1;
}
