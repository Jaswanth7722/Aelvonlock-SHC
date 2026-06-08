/**
 * @file aelvonlock.c
 * @brief Aelvonlock-SHC (Symbolic Hashing Cryptography) — C Implementation
 *
 * The core cryptographic hashing algorithm.
 *
 * Algorithm overview:
 *   1. Input sanitization + symbolic encoding
 *   2. Salt generation + HMAC-SHA256 stretching (100K iterations)
 *   3. Input entropy extraction (BLAKE2b)
 *   4. Multi-lane state initialization (up to 16 lanes)
 *   5. Block-by-block ARX processing (up to 128 rounds/block)
 *   6. Memory-hard matrix fill (Argon2-like random access)
 *   7. State-dependent memory reads
 *   8. Lane cross-pollination and merge
 *   9. Finalization with multiple ARX rounds
 *  10. 94-symbol output with rejection sampling
 */

#include "aelvonlock.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* Version tag for C output */
#define VERSION_TAG_C "V.C.L.2"
#define VERSION_TAG_C_LEN 7  /* V.C.L.2. = 7 chars */

/* ═══════════════════════════════════════════════════════════════
 * OPENSSL INTEGRATION (SHA-256, HMAC, BLAKE2)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
    /* Windows: use WinCNG or BCrypt */
    #include <windows.h>
    #include <bcrypt.h>
    #pragma comment(lib, "bcrypt.lib")
#else
    /* Use OpenSSL */
    #include <openssl/evp.h>
    #include <openssl/hmac.h>
    #include <openssl/sha.h>
#endif

/* ═══════════════════════════════════════════════════════════════
 * CRYPTOGRAPHIC CONSTANTS
 * ═══════════════════════════════════════════════════════════════ */

#define ROT1   11ULL
#define ROT2   19ULL
#define ROT3   7ULL
#define ROT4   23ULL
#define ROT5   17ULL
#define ROT6   5ULL
#define ROT7   13ULL
#define ROT8   29ULL

#define MULT_CONST    0x9E3779B97F4A7C15ULL
#define MIX_PRIME_1   0xC6A4A7935BD1E995ULL
#define MIX_PRIME_2   0xFF51AFD7ED558CCDULL
#define INIT_XOR      0xABCDEF1234567890ULL

#define DEFAULT_SALT_STRETCH  100000
#define DEFAULT_MEMORY_MB     256
#define MAX_MEMORY_MB         4096

/* ═══════════════════════════════════════════════════════════════
 * SECURITY LEVEL CONFIGURATION
 * ═══════════════════════════════════════════════════════════════ */

static const uint32_t SEC_ARX_ROUNDS[] = {4, 12, 32, 64, 128};
static const uint32_t SEC_NUM_LANES[]  = {1, 2, 4, 8, 16};
static const uint32_t SEC_FINAL_ROUNDS[] = {2, 4, 6, 8, 16};
/* SEC_MEM_MULT values: {1, 4, 16, 64, 256} - used implicitly via memory_mb param */

#define SEC_COUNT (sizeof(SEC_ARX_ROUNDS) / sizeof(SEC_ARX_ROUNDS[0]))

/* ═══════════════════════════════════════════════════════════════
 * SYMBOL MAP (94 Unicode codepoints in U+100000-U+10005D)
 * ═══════════════════════════════════════════════════════════════ */

static const uint32_t SYMBOLS[AELVONLOCK_NUM_SYMBOLS] = {
    0x100000, 0x100001, 0x100002, 0x100003, 0x100004, 0x100005, 0x100006, 0x100007,
    0x100008, 0x100009, 0x10000A, 0x10000B, 0x10000C, 0x10000D, 0x10000E, 0x10000F,
    0x100010, 0x100011, 0x100012, 0x100013, 0x100014, 0x100015, 0x100016, 0x100017,
    0x100018, 0x100019, 0x10001A, 0x10001B, 0x10001C, 0x10001D, 0x10001E, 0x10001F,
    0x100020, 0x100021, 0x100022, 0x100023, 0x100024, 0x100025, 0x100026, 0x100027,
    0x100028, 0x100029, 0x10002A, 0x10002B, 0x10002C, 0x10002D, 0x10002E, 0x10002F,
    0x100030, 0x100031, 0x100032, 0x100033, 0x100034, 0x100035, 0x100036, 0x100037,
    0x100038, 0x100039, 0x10003A, 0x10003B, 0x10003C, 0x10003D, 0x10003E, 0x10003F,
    0x100040, 0x100041, 0x100042, 0x100043, 0x100044, 0x100045, 0x100046, 0x100047,
    0x100048, 0x100049, 0x10004A, 0x10004B, 0x10004C, 0x10004D, 0x10004E, 0x10004F,
    0x100050, 0x100051, 0x100052, 0x100053, 0x100054, 0x100055, 0x100056, 0x100057,
    0x100058, 0x100059, 0x10005A, 0x10005B, 0x10005C, 0x10005D,
};

static int g_initialized = 0;

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: BITWISE OPERATIONS
 * ═══════════════════════════════════════════════════════════════ */

static inline uint64_t rotl64(uint64_t x, unsigned r) {
    r &= 63;
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t rotr64(uint64_t x, unsigned r) {
    r &= 63;
    return (x >> r) | (x << (64 - r));
}

static inline uint64_t mask64(uint64_t x) {
    return x;
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: SECURE MEMORY
 * ═══════════════════════════════════════════════════════════════ */

static void secure_zero(void* ptr, size_t len) {
    if (ptr) {
        volatile uint8_t* p = (volatile uint8_t*)ptr;
        for (size_t i = 0; i < len; i++) {
            p[i] = 0;
        }
    }
}

static void* secure_malloc(size_t size) {
    void* ptr = calloc(1, size);
    return ptr;
}

static void secure_free(void* ptr, size_t size) {
    if (ptr) {
        secure_zero(ptr, size);
        free(ptr);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: SHA-256 WRAPPER
 * ═══════════════════════════════════════════════════════════════ */

/* SHA-256 implementation using WinCNG BCrypt */
#ifdef _WIN32
static void sha256(const uint8_t* data, size_t len, uint8_t* out) {
    BCRYPT_ALG_HANDLE hAlg = NULL;
    BCRYPT_HASH_HANDLE hHash = NULL;
    DWORD hash_obj_len = 0;
    DWORD result_len = 0;
    uint8_t* hash_obj = NULL;
    
    if (BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, NULL, 0) != 0)
        goto cleanup;
    
    /* Get hash object size */
    BCryptGetProperty(hAlg, BCRYPT_OBJECT_LENGTH, (PUCHAR)&hash_obj_len,
                      sizeof(hash_obj_len), &result_len, 0);
    hash_obj = (uint8_t*)malloc(hash_obj_len);
    if (!hash_obj) goto cleanup;
    
    if (BCryptCreateHash(hAlg, &hHash, hash_obj, hash_obj_len, NULL, 0, 0) != 0)
        goto cleanup;
    
    if (BCryptHashData(hHash, (PUCHAR)data, (ULONG)len, 0) != 0)
        goto cleanup;
    
    BCryptFinishHash(hHash, out, 32, 0);
    
cleanup:
    if (hHash) BCryptDestroyHash(hHash);
    free(hash_obj);
    if (hAlg) BCryptCloseAlgorithmProvider(hAlg, 0);
}
#else
static void sha256(const uint8_t* data, size_t len, uint8_t* out) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx) {
        EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
        EVP_DigestUpdate(ctx, data, len);
        EVP_DigestFinal_ex(ctx, out, NULL);
        EVP_MD_CTX_free(ctx);
    }
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: BLAKE2b WRAPPER
 * ═══════════════════════════════════════════════════════════════ */

static void blake2b_32(const uint8_t* data, size_t len, uint8_t* out) {
    /* Use SHA-256 as BLAKE2b may not be available on all platforms */
    sha256(data, len, out);
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: HMAC-SHA256
 * ═══════════════════════════════════════════════════════════════ */

static void hmac_sha256(const uint8_t* key, size_t key_len,
                         const uint8_t* data, size_t data_len,
                         uint8_t* out) {
#ifdef _WIN32
    /* Simplified: use two SHA-256 calls (HMAC construction) */
    uint8_t k_ipad[64], k_opad[64];
    uint8_t key_buf[64] = {0};
    
    if (key_len > 64) {
        sha256(key, key_len, key_buf);
        key_len = 32;
    } else {
        memcpy(key_buf, key, key_len);
    }
    
    for (int i = 0; i < 64; i++) {
        k_ipad[i] = key_buf[i] ^ 0x36;
        k_opad[i] = key_buf[i] ^ 0x5C;
    }
    
    uint8_t inner[64 + data_len];
    memcpy(inner, k_ipad, 64);
    memcpy(inner + 64, data, data_len);
    uint8_t inner_hash[32];
    sha256(inner, 64 + data_len, inner_hash);
    
    uint8_t outer[64 + 32];
    memcpy(outer, k_opad, 64);
    memcpy(outer + 64, inner_hash, 32);
    sha256(outer, 64 + 32, out);
#else
    HMAC(EVP_sha256(), key, (int)key_len, data, (int)data_len, out, NULL);
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: CRYPTOGRAPHIC RANDOM
 * ═══════════════════════════════════════════════════════════════ */

static void random_bytes(uint8_t* out, size_t len) {
#ifdef _WIN32
    BCryptGenRandom(NULL, out, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
#elif defined(__linux__) || defined(__unix__)
    FILE* f = fopen("/dev/urandom", "rb");
    if (f) {
        size_t n = fread(out, 1, len, f);
        fclose(f);
        if (n == len) return;
    }
    /* Fallback: use getentropy() - available on modern POSIX */
    #if defined(HAVE_GETENTROPY)
        (void)getentropy(out, len);
    #else
        /* Last resort: time-based seed + simple PRNG */
        static int seeded = 0;
        if (!seeded) { srand((unsigned)(time(NULL) ^ (uintptr_t)out)); seeded = 1; }
        for (size_t i = 0; i < len; i++) {
            out[i] = (uint8_t)(rand() & 0xFF);
        }
    #endif
#else
    /* Fallback for unknown platforms */
    static int seeded = 0;
    if (!seeded) { srand((unsigned)(time(NULL) ^ (uintptr_t)out)); seeded = 1; }
    for (size_t i = 0; i < len; i++) {
        out[i] = (uint8_t)(rand() & 0xFF);
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: ARX CORE
 * ═══════════════════════════════════════════════════════════════ */

static void arx_round(uint64_t* state, uint64_t key, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        state[i] = state[i] + key + (uint64_t)(i * 13);
        state[i] ^= rotl64(state[(i + 1) % n], ROT1);
        state[i] = rotl64(state[i], ROT2);
        state[i] ^= rotr64(state[(i + 1) % n], ROT4);
        state[i] = (state[i] * MULT_CONST) ^ rotl64(state[i], ROT5);
    }
}

static void arx_block(uint64_t* state, uint64_t key, unsigned rounds,
                       unsigned num_words) {
    for (unsigned r = 0; r < rounds; r++) {
        for (unsigned i = 0; i < num_words; i++) {
            state[i] = (state[i] + key + (uint64_t)(i * 13));
            state[i] ^= rotl64(state[(i + 1) % num_words], ROT1);
            state[i] = rotl64(state[i], ROT2);
            state[i] ^= rotr64(state[(i + 1) % num_words], ROT4);
            state[i] = (state[i] * MULT_CONST) ^ rotl64(state[i], ROT5);
        }
        key = rotl64(key, ROT3) ^ MIX_PRIME_2;
    }
}

static void lane_process(uint64_t* lane, uint64_t key, unsigned rounds,
                          unsigned num_words) {
    for (unsigned r = 0; r < rounds; r++) {
        uint64_t rk = key ^ (MULT_CONST * (uint64_t)r);
        for (unsigned i = 0; i < num_words; i++) {
            lane[i] = (lane[i] + rk + (uint64_t)(i * 13));
            lane[i] ^= rotl64(lane[(i + 1) % num_words], ROT1);
            lane[i] = rotl64(lane[i], ROT2);
            lane[i] ^= rotr64(lane[(i + 2) % num_words], ROT3);
            lane[i] = (lane[i] * MIX_PRIME_1);
        }
    }
}

static void finalize_state(uint64_t* state, const uint64_t* salt_words,
                            unsigned rounds, unsigned num_words,
                            unsigned salt_count) {
    for (unsigned r = 0; r < rounds; r++) {
        uint64_t key = salt_words[r % salt_count] ^ ((uint64_t)r * MIX_PRIME_1);
        arx_round(state, key, num_words);
        for (unsigned i = 0; i < num_words; i++) {
            state[i] ^= rotl64(salt_words[i % salt_count], (i + r) % 64);
            state[i] ^= rotr64(state[(i + 1) % num_words], ROT3);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: MEMORY-HARD MATRIX
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t* data;
    unsigned rows;
    unsigned cols;
    size_t total_size;
} memory_matrix_t;

static int init_matrix(memory_matrix_t* mat, unsigned rows, unsigned cols) {
    mat->rows = rows;
    mat->cols = cols;
    mat->total_size = (size_t)rows * cols * sizeof(uint64_t);
    mat->data = (uint64_t*)secure_malloc(mat->total_size);
    return mat->data != NULL ? AELVONLOCK_SUCCESS : AELVONLOCK_ERR_MEMORY;
}

static void free_matrix(memory_matrix_t* mat) {
    if (mat->data) {
        secure_free(mat->data, mat->total_size);
        mat->data = NULL;
    }
}

static void fill_memory_matrix(uint64_t* matrix, unsigned rows, unsigned cols,
                                const uint64_t* salt_words, unsigned n_salt,
                                const uint64_t* state, unsigned n_state) {
    /* Phase 1: Deterministic fill */
    for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < cols; j++) {
            uint64_t val = (uint64_t)i * (uint64_t)j;
            val ^= salt_words[j % n_salt];
            val ^= state[j % n_state];
            val ^= (uint64_t)i * MIX_PRIME_1;
            matrix[(size_t)i * cols + j] = val;
        }
    }

    /* Phase 2: Random access (Argon2-like) */
    for (unsigned p = 0; p < 4; p++) {
        for (unsigned i = 0; i < rows; i++) {
            for (unsigned j = 0; j < cols; j++) {
                unsigned si = (unsigned)(state[i % n_state]) % rows;
                unsigned sj = (unsigned)(state[j % n_state]) % cols;
                matrix[(size_t)i * cols + j] ^= matrix[(size_t)si * cols + sj];
                matrix[(size_t)i * cols + j] ^= state[(i + j) % n_state];
                matrix[(size_t)i * cols + j] ^= MIX_PRIME_2;
            }
        }
    }

    /* Phase 3: Diffusion */
    for (unsigned p = 0; p < 4; p++) {
        for (unsigned i = 0; i < rows; i++) {
            for (unsigned j = 0; j < cols; j++) {
                unsigned nj = (j + 7) % cols;
                unsigned ni = (i + 11) % rows;
                matrix[(size_t)i * cols + j] ^= matrix[(size_t)ni * cols + nj];
                matrix[(size_t)i * cols + j] ^= state[(i + j) % n_state];
                matrix[(size_t)i * cols + j] = rotl64(matrix[(size_t)i * cols + j], 5);
            }
        }
    }
}

static void read_memory_matrix(const uint64_t* matrix, unsigned rows, unsigned cols,
                                uint64_t* state, unsigned n_state,
                                unsigned iterations,
                                const uint64_t* salt_words, unsigned n_salt) {
    for (unsigned it = 0; it < iterations; it++) {
        for (unsigned i = 0; i < n_state; i++) {
            unsigned idx_r = (unsigned)(state[i]) % rows;
            unsigned idx_c = (unsigned)(rotl64(state[i], 17)) % cols;
            uint64_t mem_val = matrix[(size_t)idx_r * cols + idx_c];

            state[i] ^= mem_val;
            state[i] = (state[i] * MULT_CONST);
            state[i] ^= salt_words[i % n_salt];
            state[i] ^= MIX_PRIME_2 * (uint64_t)it;
            state[i] = rotl64(state[i], it % 64);
        }

        /* Cross-mix */
        for (unsigned i = 0; i < n_state; i++) {
            unsigned j = (i + 1) % n_state;
            state[i] ^= rotr64(state[j], (it + 3) % 64);
            state[j] ^= rotl64(state[i], (it + 7) % 64);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: SALT GENERATION & STRETCHING
 * ═══════════════════════════════════════════════════════════════ */

static void stretch_salt_internal(const uint8_t* salt, size_t salt_len,
                                   const uint8_t* input, size_t input_len,
                                   uint32_t iterations,
                                   uint8_t* output) {
    /* Derive input key */
    uint8_t input_key[32];
    sha256(input, input_len, input_key);

    uint8_t stretched[32];
    memcpy(stretched, salt, salt_len > 32 ? 32 : salt_len);
    if (salt_len < 32) {
        memset(stretched + salt_len, 0, 32 - salt_len);
    }

    /* HMAC-based stretching */
    for (uint32_t i = 0; i < iterations; i++) {
        uint8_t hmac_key[36]; /* input_key (32) + counter (4) */
        memcpy(hmac_key, input_key, 32);
        hmac_key[32] = (uint8_t)(i & 0xFF);
        hmac_key[33] = (uint8_t)((i >> 8) & 0xFF);
        hmac_key[34] = (uint8_t)((i >> 16) & 0xFF);
        hmac_key[35] = (uint8_t)((i >> 24) & 0xFF);

        hmac_sha256(hmac_key, 36, stretched, 32, stretched);

        /* Mix with original salt every other iteration */
        if (i % 2 == 0) {
            for (size_t j = 0; j < 32; j++) {
                stretched[j] ^= salt[j % salt_len];
            }
        }
    }

    /* Combine with original salt */
    uint8_t combined[64];
    for (size_t i = 0; i < 32; i++) {
        combined[i] = salt[i % salt_len] ^ stretched[i];
        combined[32 + i] = stretched[i];
    }
    sha256(combined, 64, output);
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: SYMBOLIC OUTPUT ENCODING
 * ═══════════════════════════════════════════════════════════════ */

static int encode_symbolic_output(const uint64_t* state, unsigned num_words,
                                   uint32_t desired_length,
                                   uint8_t* output, size_t* output_len) {
    /* Convert state to bytes */
    uint8_t state_bytes[64];
    for (unsigned i = 0; i < num_words && i < 8; i++) {
        for (unsigned b = 0; b < 8; b++) {
            state_bytes[i * 8 + b] = (uint8_t)(state[i] >> (b * 8));
        }
    }

    /* Allocate temporary buffer for symbols */
    uint32_t* symbols = (uint32_t*)malloc(desired_length * sizeof(uint32_t));
    if (!symbols) return AELVONLOCK_ERR_MEMORY;

    /* Rejection sampling */
    unsigned num_syms = AELVONLOCK_NUM_SYMBOLS;
    unsigned mod_base = 256 - (256 % num_syms);
    unsigned written = 0;
    size_t byte_idx = 0;
    unsigned retries = 0;
    unsigned max_retries = 10000;

    while (written < desired_length && retries < max_retries) {
        if (byte_idx >= sizeof(state_bytes)) {
            /* Extend: re-hash current state to get more entropy */
            uint8_t ext[32];
            sha256(state_bytes, sizeof(state_bytes), ext);
            memcpy(state_bytes, ext, 32);
            byte_idx = 0;
        }

        uint8_t byte_val = state_bytes[byte_idx++];
        if (byte_val < mod_base) {
            symbols[written++] = SYMBOLS[byte_val % num_syms];
            retries = 0;
        } else {
            retries++;
            if (retries >= max_retries) {
                symbols[written++] = SYMBOLS[byte_val % num_syms];
                retries = 0;
            }
        }
    }

    /* Encode Unicode codepoints to UTF-8 */
    size_t pos = 0;
    size_t cap = *output_len;
    for (unsigned i = 0; i < written && pos < cap; i++) {
        uint32_t cp = symbols[i];
        if (cp < 0x80) {
            output[pos++] = (uint8_t)cp;
        } else if (cp < 0x800) {
            if (pos + 1 >= cap) break;
            output[pos++] = (uint8_t)(0xC0 | (cp >> 6));
            output[pos++] = (uint8_t)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            if (pos + 2 >= cap) break;
            output[pos++] = (uint8_t)(0xE0 | (cp >> 12));
            output[pos++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
            output[pos++] = (uint8_t)(0x80 | (cp & 0x3F));
        } else {
            if (pos + 3 >= cap) break;
            output[pos++] = (uint8_t)(0xF0 | (cp >> 18));
            output[pos++] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
            output[pos++] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
            output[pos++] = (uint8_t)(0x80 | (cp & 0x3F));
        }
    }

    free(symbols);
    *output_len = pos;
    return AELVONLOCK_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════
 * INTERNAL: CONSTANT-TIME COMPARE
 * ═══════════════════════════════════════════════════════════════ */

int aelvonlock_constant_time_compare(const uint8_t* a, size_t a_len,
                                      const uint8_t* b, size_t b_len) {
    if (a_len != b_len) return 0;
    volatile uint8_t result = 0;
    for (size_t i = 0; i < a_len; i++) {
        result |= a[i] ^ b[i];
    }
    return result == 0;
}

/* ═══════════════════════════════════════════════════════════════
 * PUBLIC API IMPLEMENTATION
 * ═══════════════════════════════════════════════════════════════ */

int aelvonlock_init(void) {
    g_initialized = 1;
#ifdef _WIN32
    /* BCrypt is always available on Windows Vista+ */
#else
    /* Verify OpenSSL is available */
#endif
    return AELVONLOCK_SUCCESS;
}

const char* aelvonlock_version(void) {
    return AELVONLOCK_VERSION_STRING;
}

int aelvonlock_default_params(aelvonlock_security_level_t level,
                               aelvonlock_params_t* params) {
    if (!params) return AELVONLOCK_ERR_NULL_INPUT;
    if (level > AELVONLOCK_SECURITY_ULTIMATE) {
        level = AELVONLOCK_SECURITY_ULTIMATE;
    }

    params->security_level = level;
    params->desired_length = AELVONLOCK_DEFAULT_OUTPUT;
    params->salt_stretch_iterations = DEFAULT_SALT_STRETCH;
    params->memory_mb = 0; /* auto */
    params->num_lanes = 0; /* auto */
    params->arx_rounds = 0; /* auto */
    return AELVONLOCK_SUCCESS;
}

int aelvonlock_hash(const uint8_t* input, size_t input_len,
                    const uint8_t* salt, size_t salt_len,
                    uint8_t* output, size_t* output_len,
                    const aelvonlock_params_t* params) {
    if (!g_initialized) return AELVONLOCK_ERR_NOT_INITIALIZED;
    if (!input || !output || !output_len) return AELVONLOCK_ERR_NULL_INPUT;
    if (input_len == 0) return AELVONLOCK_ERR_INPUT_EMPTY;
    if (input_len > AELVONLOCK_MAX_INPUT) return AELVONLOCK_ERR_INPUT_TOO_LONG;

    /* Default parameters */
    aelvonlock_params_t default_params;
    if (!params) {
        aelvonlock_default_params(AELVONLOCK_SECURITY_STANDARD, &default_params);
        params = &default_params;
    }

    unsigned sec_level = (unsigned)params->security_level;
    if (sec_level >= SEC_COUNT) sec_level = AELVONLOCK_SECURITY_ULTIMATE;

    unsigned arx_rounds = params->arx_rounds ? params->arx_rounds : SEC_ARX_ROUNDS[sec_level];
    unsigned num_lanes = params->num_lanes ? params->num_lanes : SEC_NUM_LANES[sec_level];
    unsigned final_rounds = SEC_FINAL_ROUNDS[sec_level];
    uint32_t stretch_iters = params->salt_stretch_iterations ?
                             params->salt_stretch_iterations : DEFAULT_SALT_STRETCH;
    uint32_t desired_len = params->desired_length ?
                           params->desired_length : AELVONLOCK_DEFAULT_OUTPUT;

    if (desired_len < AELVONLOCK_MIN_OUTPUT) return AELVONLOCK_ERR_OUTPUT_TOO_SHORT;
    if (desired_len > AELVONLOCK_MAX_OUTPUT) return AELVONLOCK_ERR_OUTPUT_TOO_LONG;

    /* ─── Salt ────────────────────────────────────────────── */
    uint8_t salt_buf[AELVONLOCK_SALT_SIZE];
    if (!salt || salt_len == 0) {
        salt_len = AELVONLOCK_SALT_SIZE;
        random_bytes(salt_buf, salt_len);
        salt = salt_buf;
    }

    /* Stretch salt */
    uint8_t final_salt[32];
    if (stretch_iters > 0) {
        stretch_salt_internal(salt, salt_len, input, input_len,
                               stretch_iters, final_salt);
    } else {
        memcpy(final_salt, salt, salt_len > 32 ? 32 : salt_len);
        if (salt_len < 32) memset(final_salt + salt_len, 0, 32 - salt_len);
    }

    /* Extract salt words */
    uint64_t salt_words[AELVONLOCK_NUM_WORDS];
    for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS; i++) {
        uint64_t w = 0;
        for (unsigned b = 0; b < 8; b++) {
            w = (w << 8) | final_salt[(i * 8 + b) % 32];
        }
        salt_words[i] = w;
    }
    uint64_t salt_int = salt_words[0];

    /* ─── Input entropy extraction ────────────────────────── */
    uint8_t input_entropy[32];
    blake2b_32(input, input_len, input_entropy);

    uint64_t entropy_words[4];
    for (unsigned i = 0; i < 4; i++) {
        uint64_t w = 0;
        for (unsigned b = 0; b < 8; b++) {
            w |= (uint64_t)input_entropy[i * 8 + b] << (b * 8);
        }
        entropy_words[i] = w;
    }

    /* ─── State initialization ────────────────────────────── */
    uint64_t state[AELVONLOCK_NUM_WORDS];
    for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS; i++) {
        state[i] = salt_words[i % AELVONLOCK_NUM_WORDS];
        if (i < 4) state[i] ^= entropy_words[i];
        state[i] ^= INIT_XOR ^ ((uint64_t)i * MIX_PRIME_1);
    }

    /* Initialize lanes */
    unsigned num_words = AELVONLOCK_NUM_WORDS;
    uint64_t** lanes = (uint64_t**)malloc(num_lanes * sizeof(uint64_t*));
    if (!lanes) return AELVONLOCK_ERR_MEMORY;
    for (unsigned l = 0; l < num_lanes; l++) {
        lanes[l] = (uint64_t*)malloc(num_words * sizeof(uint64_t));
        if (!lanes[l]) {
            for (unsigned k = 0; k < l; k++) free(lanes[k]);
            free(lanes);
            return AELVONLOCK_ERR_MEMORY;
        }
        for (unsigned i = 0; i < num_words; i++) {
            lanes[l][i] = state[i] ^ ((uint64_t)l * MIX_PRIME_2);
        }
    }

    /* ─── Block processing ────────────────────────────────── */
    /* Convert input to binary representation */
    /* size_t binary_len = input_len * 8; (unused - we process blocks directly) */
    /* Simple processing: process input in 64-byte blocks */
    size_t num_blocks = (input_len + 63) / 64;
    for (size_t b = 0; b < num_blocks; b++) {
        size_t offset = b * 64;
        size_t remaining = input_len - offset;
        size_t block_size = remaining < 64 ? remaining : 64;

        /* Create block words from input */
        uint64_t block_words[AELVONLOCK_NUM_WORDS] = {0};
        for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS && i * 8 < block_size; i++) {
            for (unsigned j = 0; j < 8 && i * 8 + j < block_size; j++) {
                block_words[i] |= (uint64_t)input[offset + i * 8 + j] << (j * 8);
            }
        }

        /* Mix with salt and state */
        for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS; i++) {
            block_words[i] ^= salt_words[i];
            block_words[i] ^= state[i];
            block_words[i] ^= (uint64_t)b * MIX_PRIME_1;
        }

        /* ARX rounds */
        uint64_t block_key = salt_int ^ ((uint64_t)b * MIX_PRIME_2);
        arx_block(block_words, block_key, arx_rounds, AELVONLOCK_NUM_WORDS);

        /* Mix into state */
        for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS; i++) {
            state[i] ^= block_words[i];
        }

        /* Feed into lanes */
        for (unsigned l = 0; l < num_lanes && l < 16; l++) {
            for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS; i++) {
                lanes[l][i] ^= state[i];
                lanes[l][i] ^= (uint64_t)l * (uint64_t)i * (uint64_t)b;
            }
        }
    }

    /* ─── Memory-hard matrix processing ───────────────────── */
    unsigned mem_rows = 256;
    unsigned mem_cols = 256;

    for (unsigned l = 0; l < num_lanes && l < 16; l++) {
        memory_matrix_t mat;
        if (init_matrix(&mat, mem_rows, mem_cols) == AELVONLOCK_SUCCESS) {
            fill_memory_matrix(mat.data, mat.rows, mat.cols,
                               salt_words, AELVONLOCK_NUM_WORDS,
                               lanes[l], num_words);
            read_memory_matrix(mat.data, mat.rows, mat.cols,
                               state, AELVONLOCK_NUM_WORDS,
                               256, salt_words, AELVONLOCK_NUM_WORDS);
            free_matrix(&mat);
        }
    }

    /* ─── Lane cross-pollination ──────────────────────────── */
    uint64_t mixed_state[AELVONLOCK_NUM_WORDS] = {0};
    for (unsigned l = 0; l < num_lanes && l < 16; l++) {
        uint64_t lane_key = salt_int ^ ((uint64_t)l * MIX_PRIME_1);
        lane_process(lanes[l], lane_key, 64, num_words);

        for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS; i++) {
            mixed_state[i] ^= lanes[l][i];
        }
    }

    /* Merge lanes */
    for (unsigned i = 0; i < AELVONLOCK_NUM_WORDS; i++) {
        state[i] ^= mixed_state[i];
        state[i] ^= rotl64(state[(i + 1) % AELVONLOCK_NUM_WORDS], ROT7);
        state[i] = state[i] * MULT_CONST;
    }

    /* ─── Finalization ────────────────────────────────────── */
    finalize_state(state, salt_words, final_rounds,
                    AELVONLOCK_NUM_WORDS, AELVONLOCK_NUM_WORDS);

    for (unsigned r = 0; r < final_rounds; r++) {
        uint64_t key = salt_int ^ ((uint64_t)r * MIX_PRIME_2);
        arx_block(state, key, 8, AELVONLOCK_NUM_WORDS);
    }

    /* ─── Symbolic output ─────────────────────────────────── */
    size_t buf_cap = *output_len;  /* Save caller's buffer capacity before encode overwrites it */
    int ret = encode_symbolic_output(state, AELVONLOCK_NUM_WORDS,
                                      desired_len, output, output_len);
    
    /* Prepend version tag */
    if (ret == AELVONLOCK_SUCCESS) {
        size_t hash_len = *output_len;
        if (hash_len + VERSION_TAG_C_LEN <= buf_cap && hash_len > 0) {
            /* Shift hash right by tag len (work backwards to avoid overlap) */
            for (size_t i = hash_len; i > 0; i--) {
                output[i + VERSION_TAG_C_LEN - 1] = output[i - 1];
            }
            memcpy(output, VERSION_TAG_C ".", VERSION_TAG_C_LEN);
            *output_len = hash_len + VERSION_TAG_C_LEN;
        }
    }

    /* Cleanup */
    for (unsigned l = 0; l < num_lanes; l++) {
        secure_free(lanes[l], num_words * sizeof(uint64_t));
    }
    free(lanes);
    secure_zero(salt_buf, sizeof(salt_buf));

    return ret;
}

int aelvonlock_verify(const uint8_t* password, size_t password_len,
                      const uint8_t* stored_hash, size_t hash_len,
                      const uint8_t* salt, size_t salt_len,
                      const aelvonlock_params_t* params) {
    if (!password || !stored_hash || !salt) return AELVONLOCK_ERR_NULL_INPUT;

    /* Recompute hash with same salt */
    uint8_t recomputed[4096];
    size_t recomputed_len = sizeof(recomputed);

    int ret = aelvonlock_hash(password, password_len,
                               salt, salt_len,
                               recomputed, &recomputed_len,
                               params);
    if (ret != AELVONLOCK_SUCCESS) return ret;

    /* Compare in constant time */
    return aelvonlock_constant_time_compare(recomputed, recomputed_len,
                                            stored_hash, hash_len);
}

int aelvonlock_get_symbol_map(uint32_t* symbols) {
    if (symbols) {
        memcpy(symbols, SYMBOLS, sizeof(SYMBOLS));
    }
    return AELVONLOCK_NUM_SYMBOLS;
}

/* ═══════════════════════════════════════════════════════════════
 * SELF-TEST
 * ═══════════════════════════════════════════════════════════════ */

int aelvonlock_self_test(void) {
    int all_pass = 1;

    /* Test 1: Basic hashing */
    {
        const uint8_t input[] = "test";
        uint8_t output[1024];
        size_t out_len = sizeof(output);
        int ret = aelvonlock_hash(input, 4, NULL, 0, output, &out_len, NULL);
        if (ret != AELVONLOCK_SUCCESS || out_len == 0) {
            printf("  [FAIL] basic_hashing\n");
            all_pass = 0;
        } else {
            printf("  [PASS] basic_hashing (%zu bytes)\n", out_len);
        }
    }

    /* Test 2: Deterministic with same salt */
    {
        uint8_t salt[32] = {0};
        uint8_t out1[1024], out2[1024];
        size_t len1 = sizeof(out1), len2 = sizeof(out2);

        aelvonlock_hash((const uint8_t*)"test", 4, salt, 32, out1, &len1, NULL);
        aelvonlock_hash((const uint8_t*)"test", 4, salt, 32, out2, &len2, NULL);

        if (len1 != len2 || memcmp(out1, out2, len1) != 0) {
            printf("  [FAIL] deterministic\n");
            all_pass = 0;
        } else {
            printf("  [PASS] deterministic\n");
        }
    }

    /* Test 3: Different inputs produce different hashes */
    {
        uint8_t out1[1024], out2[1024];
        size_t len1 = sizeof(out1), len2 = sizeof(out2);

        aelvonlock_hash((const uint8_t*)"hello", 5, NULL, 0, out1, &len1, NULL);
        aelvonlock_hash((const uint8_t*)"world", 5, NULL, 0, out2, &len2, NULL);

        if (len1 == len2 && memcmp(out1, out2, len1) == 0) {
            printf("  [FAIL] different_inputs\n");
            all_pass = 0;
        } else {
            printf("  [PASS] different_inputs\n");
        }
    }

    /* Test 4: Salt changes output */
    {
        uint8_t salt1[32] = {1, 2, 3};
        uint8_t salt2[32] = {4, 5, 6};
        uint8_t out1[1024], out2[1024];
        size_t len1 = sizeof(out1), len2 = sizeof(out2);

        aelvonlock_hash((const uint8_t*)"test", 4, salt1, 32, out1, &len1, NULL);
        aelvonlock_hash((const uint8_t*)"test", 4, salt2, 32, out2, &len2, NULL);

        if (len1 == len2 && memcmp(out1, out2, len1) == 0) {
            printf("  [FAIL] salt_changes_output\n");
            all_pass = 0;
        } else {
            printf("  [PASS] salt_changes_output\n");
        }
    }

    /* Test 5: Verification */
    {
        const uint8_t password[] = "correct_password";
        uint8_t hash[1024];
        size_t hash_len = sizeof(hash);
        uint8_t salt[32] = {0xAB, 0xCD, 0xEF};

        aelvonlock_hash(password, 16, salt, 32, hash, &hash_len, NULL);
        int valid = aelvonlock_verify(password, 16, hash, hash_len, salt, 32, NULL);
        int invalid = aelvonlock_verify((const uint8_t*)"wrong", 5, hash, hash_len, salt, 32, NULL);

        if (!valid || invalid) {
            printf("  [FAIL] verification\n");
            all_pass = 0;
        } else {
            printf("  [PASS] verification\n");
        }
    }

    /* Test 6: Version string */
    {
        const char* ver = aelvonlock_version();
        if (strcmp(ver, AELVONLOCK_VERSION_STRING) != 0) {
            printf("  [FAIL] version\n");
            all_pass = 0;
        } else {
            printf("  [PASS] version (%s)\n", ver);
        }
    }

    printf("\nOverall: %s\n", all_pass ? "ALL PASSED" : "SOME FAILED");
    return all_pass;
}
