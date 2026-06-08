/**
 * @file aelvonlock.h
 * @brief Aelvonlock-SHC (Symbolic Hashing Cryptography) — C Implementation
 *
 * Production-grade cryptographic hashing library.
 * Provides memory-hard, symbolic-output hashing with configurable security levels.
 *
 * Algorithm: Aelvonlock-SHC v2.0
 * - ARX (Addition-Rotation-XOR) core with 128 rounds maximum
 * - 16 parallel processing lanes with cross-pollination
 * - Input-dependent memory-hard matrix (Argon2-like)
 * - HMAC-SHA256 salt stretching (100K iterations)
 * - 94-symbol Unicode output with rejection sampling
 * - Constant-time comparison
 *
 * @version 2.0.0
 * @author Aelvonlock Developers
 * @license MIT
 */
#ifndef AELVONLOCK_H
#define AELVONLOCK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * VERSION
 * ═══════════════════════════════════════════════════════════════ */

#define AELVONLOCK_VERSION_MAJOR 2
#define AELVONLOCK_VERSION_MINOR 0
#define AELVONLOCK_VERSION_PATCH 0
#define AELVONLOCK_VERSION_STRING "2.0.0"
#define AELVONLOCK_VERSION_TAG "AEON-2.0"

/* ═══════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════ */

#define AELVONLOCK_WORD_SIZE      64
#define AELVONLOCK_BLOCK_SIZE     512
#define AELVONLOCK_NUM_WORDS      8
#define AELVONLOCK_MAX_INPUT      (1024 * 1024)  /* 1 MB */
#define AELVONLOCK_MIN_OUTPUT     8
#define AELVONLOCK_MAX_OUTPUT     4096
#define AELVONLOCK_DEFAULT_OUTPUT 64
#define AELVONLOCK_SALT_SIZE      32
#define AELVONLOCK_NUM_SYMBOLS    94

/* ═══════════════════════════════════════════════════════════════
 * SECURITY LEVELS
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    AELVONLOCK_SECURITY_LOW       = 0,  /* 4 rounds, 1 lane */
    AELVONLOCK_SECURITY_STANDARD  = 1,  /* 12 rounds, 2 lanes */
    AELVONLOCK_SECURITY_HIGH      = 2,  /* 32 rounds, 4 lanes */
    AELVONLOCK_SECURITY_MAXIMUM   = 3,  /* 64 rounds, 8 lanes */
    AELVONLOCK_SECURITY_ULTIMATE  = 4,  /* 128 rounds, 16 lanes */
} aelvonlock_security_level_t;

/* ═══════════════════════════════════════════════════════════════
 * ERROR CODES
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    AELVONLOCK_SUCCESS           = 0,
    AELVONLOCK_ERR_NULL_INPUT    = -1,
    AELVONLOCK_ERR_INPUT_TOO_LONG = -2,
    AELVONLOCK_ERR_INPUT_EMPTY   = -3,
    AELVONLOCK_ERR_OUTPUT_TOO_SHORT = -4,
    AELVONLOCK_ERR_OUTPUT_TOO_LONG  = -5,
    AELVONLOCK_ERR_MEMORY        = -6,
    AELVONLOCK_ERR_INTERNAL      = -7,
    AELVONLOCK_ERR_NOT_INITIALIZED = -8,
} aelvonlock_error_t;

/* ═══════════════════════════════════════════════════════════════
 * PARAMETERS
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    aelvonlock_security_level_t security_level;
    uint32_t desired_length;         /* Output symbol count (8-4096) */
    uint32_t salt_stretch_iterations; /* 0 = disabled, recommend 100000 */
    uint32_t memory_mb;               /* Per-lane memory limit (0 = auto) */
    uint32_t num_lanes;               /* 0 = auto based on security level */
    uint32_t arx_rounds;              /* 0 = auto based on security level */
} aelvonlock_params_t;

/* ═══════════════════════════════════════════════════════════════
 * PUBLIC API
 * ═══════════════════════════════════════════════════════════════ */

/**
 * @brief Initialize the Aelvonlock library.
 * Must be called once before any other function.
 * @return AELVONLOCK_SUCCESS on success, error code otherwise.
 */
int aelvonlock_init(void);

/**
 * @brief Get the library version string.
 * @return Version string (e.g., "2.0.0").
 */
const char* aelvonlock_version(void);

/**
 * @brief Get default parameters for a security level.
 * @param level Security level.
 * @param params Output parameters structure.
 * @return AELVONLOCK_SUCCESS on success.
 */
int aelvonlock_default_params(aelvonlock_security_level_t level,
                               aelvonlock_params_t* params);

/**
 * @brief Compute Aelvonlock hash.
 *
 * Core hashing function. Takes input text with optional salt
 * and produces symbolic hash output.
 *
 * @param input       Input data (UTF-8 encoded text).
 * @param input_len   Length of input in bytes.
 * @param salt        Optional salt bytes (NULL = auto-generated).
 * @param salt_len    Length of salt in bytes (0 = AELVONLOCK_SALT_SIZE).
 * @param output      Output buffer for hash symbols (UTF-8).
 * @param output_len  In: buffer capacity. Out: actual hash length.
 * @param params      Hash parameters (NULL = standard defaults).
 * @return AELVONLOCK_SUCCESS on success, error code otherwise.
 */
int aelvonlock_hash(const uint8_t* input, size_t input_len,
                    const uint8_t* salt, size_t salt_len,
                    uint8_t* output, size_t* output_len,
                    const aelvonlock_params_t* params);

/**
 * @brief Verify a password against a stored hash.
 *
 * Recomputes hash with the same salt and compares in constant time.
 *
 * @param password      Password to verify.
 * @param password_len  Password length.
 * @param stored_hash   Previously stored hash.
 * @param hash_len      Stored hash length.
 * @param salt          Salt used for original hash.
 * @param salt_len      Salt length.
 * @param params        Hash parameters used for original hash.
 * @return 1 if valid, 0 if invalid, negative on error.
 */
int aelvonlock_verify(const uint8_t* password, size_t password_len,
                      const uint8_t* stored_hash, size_t hash_len,
                      const uint8_t* salt, size_t salt_len,
                      const aelvonlock_params_t* params);

/**
 * @brief Constant-time memory comparison.
 *
 * Compares two byte sequences in constant time
 * to prevent timing side-channel attacks.
 *
 * @param a      First buffer.
 * @param a_len  Length of first buffer.
 * @param b      Second buffer.
 * @param b_len  Length of second buffer.
 * @return 1 if equal, 0 otherwise.
 */
int aelvonlock_constant_time_compare(const uint8_t* a, size_t a_len,
                                      const uint8_t* b, size_t b_len);

/**
 * @brief Get the symbol map.
 *
 * Returns the 94 Unicode codepoints used for symbolic output.
 * Each codepoint is a uint32_t in the range U+100000-U+10005D.
 *
 * @param symbols Output buffer for 94 codepoints (can be NULL to get count).
 * @return Number of symbols (always 94).
 */
int aelvonlock_get_symbol_map(uint32_t* symbols);

/**
 * @brief Self-test the library.
 *
 * Runs comprehensive internal tests including:
 * - Basic hashing
 * - Deterministic output
 * - Salt independence
 * - Verification correctness
 * - Avalanche effect
 *
 * @return 1 if all tests pass, 0 if any fail.
 */
int aelvonlock_self_test(void);

#ifdef __cplusplus
}
#endif

#endif /* AELVONLOCK_H */
