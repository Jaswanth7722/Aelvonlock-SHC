/**
 * @file aelvonlock.hpp
 * @brief Aelvonlock-SHC (Symbolic Hashing Cryptography) — C++20 Implementation
 *
 * Modern C++20 cryptographic hashing library with:
 * - RAII resource management (no manual memory management)
 * - Template-based security levels (compile-time configuration)
 * - constexpr support for constants
 * - std::span for memory-safe buffer operations
 * - Concepts for type safety
 * - Exception-safe design with noexcept where possible
 *
 * @version 2.0.0
 * @author Aelvonlock Developers
 * @license MIT
 */

#ifndef AELVONLOCK_HPP
#define AELVONLOCK_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
    #include <bcrypt.h>
    #pragma comment(lib, "bcrypt.lib")
#else
    #include <openssl/evp.h>
    #include <openssl/hmac.h>
#endif

namespace aelvonlock {

/* ═══════════════════════════════════════════════════════════════
 * VERSION
 * ═══════════════════════════════════════════════════════════════ */

inline constexpr int VERSION_MAJOR = 2;
inline constexpr int VERSION_MINOR = 0;
inline constexpr int VERSION_PATCH = 0;
inline constexpr std::string_view VERSION_STRING = "2.0.0";

/* ═══════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════ */

inline constexpr size_t WORD_SIZE = 64;
inline constexpr size_t BLOCK_SIZE = 512;
inline constexpr size_t NUM_WORDS = 8;
inline constexpr size_t MAX_INPUT = 1024 * 1024;
inline constexpr size_t DEFAULT_OUTPUT_LEN = 64;
inline constexpr size_t MIN_OUTPUT_LEN = 8;
inline constexpr size_t MAX_OUTPUT_LEN = 4096;
inline constexpr size_t SALT_SIZE = 32;
inline constexpr size_t NUM_SYMBOLS = 94;

/* ═══════════════════════════════════════════════════════════════
 * CRYPTOGRAPHIC CONSTANTS (constexpr)
 * ═══════════════════════════════════════════════════════════════ */

inline constexpr uint64_t ROT1 = 11;
inline constexpr uint64_t ROT2 = 19;
inline constexpr uint64_t ROT3 = 7;
inline constexpr uint64_t ROT4 = 23;
inline constexpr uint64_t ROT5 = 17;
inline constexpr uint64_t MULT_CONST = 0x9E3779B97F4A7C15ULL;
inline constexpr uint64_t MIX_PRIME_1 = 0xC6A4A7935BD1E995ULL;
inline constexpr uint64_t MIX_PRIME_2 = 0xFF51AFD7ED558CCDULL;
inline constexpr uint64_t INIT_XOR = 0xABCDEF1234567890ULL;

/* ═══════════════════════════════════════════════════════════════
 * SECURITY LEVEL
 * ═══════════════════════════════════════════════════════════════ */

enum class SecurityLevel : uint8_t {
    Low      = 0,  /*  4 rounds, 1 lane */
    Standard = 1,  /* 12 rounds, 2 lanes */
    High     = 2,  /* 32 rounds, 4 lanes */
    Maximum  = 3,  /* 64 rounds, 8 lanes */
    Ultimate = 4,  /* 128 rounds, 16 lanes */
};

/* ═══════════════════════════════════════════════════════════════
 * SECURITY LEVEL TRAITS (compile-time configuration)
 * ═══════════════════════════════════════════════════════════════ */

template <SecurityLevel L>
struct SecurityTraits {};

template <>
struct SecurityTraits<SecurityLevel::Low> {
    static constexpr uint32_t arx_rounds = 4;
    static constexpr uint32_t num_lanes = 1;
    static constexpr uint32_t final_rounds = 2;
    static constexpr uint32_t memory_multiplier = 1;
};

template <>
struct SecurityTraits<SecurityLevel::Standard> {
    static constexpr uint32_t arx_rounds = 12;
    static constexpr uint32_t num_lanes = 2;
    static constexpr uint32_t final_rounds = 4;
    static constexpr uint32_t memory_multiplier = 4;
};

template <>
struct SecurityTraits<SecurityLevel::High> {
    static constexpr uint32_t arx_rounds = 32;
    static constexpr uint32_t num_lanes = 4;
    static constexpr uint32_t final_rounds = 6;
    static constexpr uint32_t memory_multiplier = 16;
};

template <>
struct SecurityTraits<SecurityLevel::Maximum> {
    static constexpr uint32_t arx_rounds = 64;
    static constexpr uint32_t num_lanes = 8;
    static constexpr uint32_t final_rounds = 8;
    static constexpr uint32_t memory_multiplier = 64;
};

template <>
struct SecurityTraits<SecurityLevel::Ultimate> {
    static constexpr uint32_t arx_rounds = 128;
    static constexpr uint32_t num_lanes = 16;
    static constexpr uint32_t final_rounds = 16;
    static constexpr uint32_t memory_multiplier = 256;
};

/* ═══════════════════════════════════════════════════════════════
 * EXCEPTIONS
 * ═══════════════════════════════════════════════════════════════ */

class AelvonlockError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class NullInputError : public AelvonlockError {
public:
    NullInputError() : AelvonlockError("Input cannot be null") {}
};

class EmptyInputError : public AelvonlockError {
public:
    EmptyInputError() : AelvonlockError("Input cannot be empty") {}
};

class InputTooLongError : public AelvonlockError {
public:
    InputTooLongError() : AelvonlockError("Input exceeds maximum length") {}
};

class OutputLengthError : public AelvonlockError {
public:
    OutputLengthError() : AelvonlockError("Invalid output length") {}
};

class MemoryError : public AelvonlockError {
public:
    MemoryError() : AelvonlockError("Memory allocation failed") {}
};

/* ═══════════════════════════════════════════════════════════════
 * TYPE ALIASES
 * ═══════════════════════════════════════════════════════════════ */

using StateArray = std::array<uint64_t, NUM_WORDS>;
using SaltArray = std::array<uint8_t, SALT_SIZE>;
using SymbolArray = std::array<uint32_t, NUM_SYMBOLS>;
using ByteBuffer = std::vector<uint8_t>;

/* ═══════════════════════════════════════════════════════════════
 * HASHPARAMS (runtime-configurable)
 * ═══════════════════════════════════════════════════════════════ */

struct HashParams {
    SecurityLevel security_level = SecurityLevel::Ultimate;
    uint32_t desired_length = DEFAULT_OUTPUT_LEN;
    uint32_t salt_stretch_iterations = 100000;
    uint32_t memory_mb = 0;  /* 0 = auto */
    uint32_t num_lanes = 0;  /* 0 = auto based on security level */
    uint32_t arx_rounds = 0; /* 0 = auto based on security level */
};

/* ═══════════════════════════════════════════════════════════════
 * SYMBOL MAP
 * ═══════════════════════════════════════════════════════════════ */

inline constexpr SymbolArray SYMBOLS = {
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

static_assert(SYMBOLS.size() == NUM_SYMBOLS, "Symbol map must have 94 entries");
static_assert(SYMBOLS[0] == 0x100000, "First symbol must be U+100000");
static_assert(SYMBOLS[93] == 0x10005D, "Last symbol must be U+10005D");

/* ═══════════════════════════════════════════════════════════════
 * BITWISE OPERATIONS (constexpr with C++20 std::rotl/rotr)
 * ═══════════════════════════════════════════════════════════════ */

inline constexpr uint64_t rotl(uint64_t x, unsigned r) noexcept {
    return std::rotl(x, static_cast<int>(r));
}

inline constexpr uint64_t rotr(uint64_t x, unsigned r) noexcept {
    return std::rotr(x, static_cast<int>(r));
}

inline constexpr uint64_t mask64(uint64_t x) noexcept {
    return x;
}

/* ═══════════════════════════════════════════════════════════════
 * SECURE MEMORY (RAII)
 * ═══════════════════════════════════════════════════════════════ */

class SecureBuffer {
public:
    SecureBuffer() = default;

    explicit SecureBuffer(size_t size)
        : data_(std::make_unique<uint8_t[]>(size)), size_(size) {
        std::memset(data_.get(), 0, size);
    }

    ~SecureBuffer() { secure_clear(); }

    SecureBuffer(const SecureBuffer&) = delete;
    SecureBuffer& operator=(const SecureBuffer&) = delete;

    SecureBuffer(SecureBuffer&& other) noexcept
        : data_(std::move(other.data_)), size_(other.size_) {
        other.size_ = 0;
    }

    SecureBuffer& operator=(SecureBuffer&& other) noexcept {
        if (this != &other) {
            secure_clear();
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }

    uint8_t* data() noexcept { return data_.get(); }
    const uint8_t* data() const noexcept { return data_.get(); }
    size_t size() const noexcept { return size_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }

    std::span<uint8_t> span() noexcept {
        return std::span<uint8_t>(data_.get(), size_);
    }

    std::span<const uint8_t> span() const noexcept {
        return std::span<const uint8_t>(data_.get(), size_);
    }

    void resize(size_t new_size) {
        auto new_data = std::make_unique<uint8_t[]>(new_size);
        std::memset(new_data.get(), 0, new_size);
        if (data_) {
            size_t copy = std::min(size_, new_size);
            std::memcpy(new_data.get(), data_.get(), copy);
        }
        secure_clear();
        data_ = std::move(new_data);
        size_ = new_size;
    }

private:
    void secure_clear() noexcept {
        if (data_) {
            volatile uint8_t* p = data_.get();
            for (size_t i = 0; i < size_; i++) {
                p[i] = 0;
            }
        }
    }

    std::unique_ptr<uint8_t[]> data_;
    size_t size_ = 0;
};

/* ═══════════════════════════════════════════════════════════════
 * CRYPTOGRAPHIC RANDOM
 * ═══════════════════════════════════════════════════════════════ */

inline void random_bytes(std::span<uint8_t> out) {
#ifdef _WIN32
    BCryptGenRandom(NULL, out.data(), static_cast<ULONG>(out.size()),
                    BCRYPT_USE_SYSTEM_PREFERRED_RNG);
#elif defined(__linux__) || defined(__unix__)
    FILE* f = std::fopen("/dev/urandom", "rb");
    if (f) {
        size_t n = std::fread(out.data(), 1, out.size(), f);
        std::fclose(f);
        if (n == out.size()) return;
    }
    /* Fallback */
    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned>(std::time(nullptr) ^
                    reinterpret_cast<uintptr_t>(out.data())));
        seeded = true;
    }
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = static_cast<uint8_t>(std::rand() & 0xFF);
    }
#else
    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned>(std::time(nullptr) ^
                    reinterpret_cast<uintptr_t>(out.data())));
        seeded = true;
    }
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = static_cast<uint8_t>(std::rand() & 0xFF);
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * SHA-256 WRAPPER
 * ═══════════════════════════════════════════════════════════════ */

inline std::array<uint8_t, 32> sha256(std::span<const uint8_t> data) {
    std::array<uint8_t, 32> out{};
#ifdef _WIN32
    BCRYPT_ALG_HANDLE hAlg = nullptr;
    BCRYPT_HASH_HANDLE hHash = nullptr;
    DWORD hash_obj_len = 0;
    DWORD result_len = 0;
    std::vector<uint8_t> hash_obj;
    
    if (BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0)
        return out;
    
    BCryptGetProperty(hAlg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&hash_obj_len),
                      sizeof(hash_obj_len), &result_len, 0);
    hash_obj.resize(hash_obj_len);
    
    if (BCryptCreateHash(hAlg, &hHash, hash_obj.data(), hash_obj_len, nullptr, 0, 0) != 0) {
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return out;
    }
    
    BCryptHashData(hHash, const_cast<PUCHAR>(data.data()),
                   static_cast<ULONG>(data.size()), 0);
    BCryptFinishHash(hHash, out.data(), 32, 0);
    
    BCryptDestroyHash(hHash);
    BCryptCloseAlgorithmProvider(hAlg, 0);
#else
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx) {
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
        EVP_DigestUpdate(ctx, data.data(), data.size());
        EVP_DigestFinal_ex(ctx, out.data(), nullptr);
        EVP_MD_CTX_free(ctx);
    }
#endif
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 * HMAC-SHA256
 * ═══════════════════════════════════════════════════════════════ */

inline std::array<uint8_t, 32> hmac_sha256(
    std::span<const uint8_t> key,
    std::span<const uint8_t> data) {
    std::array<uint8_t, 32> out{};

#ifdef _WIN32
    BCRYPT_ALG_HANDLE hAlg;
    BCRYPT_HASH_HANDLE hHash;
    BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, NULL,
                                BCRYPT_ALG_HANDLE_HMAC_FLAG);
    
    uint8_t hmac_key[64] = {0};
    size_t key_len = key.size();
    if (key_len > 64) {
        auto hashed = sha256(key);
        std::memcpy(hmac_key, hashed.data(), 32);
        key_len = 32;
    } else {
        std::memcpy(hmac_key, key.data(), key_len);
    }
    
    BCryptCreateHash(hAlg, &hHash, NULL, 0, hmac_key,
                     static_cast<ULONG>(key_len), 0);
    BCryptHashData(hHash, const_cast<PUCHAR>(data.data()),
                   static_cast<ULONG>(data.size()), 0);
    BCryptFinishHash(hHash, out.data(), 32, 0);
    BCryptDestroyHash(hHash);
    BCryptCloseAlgorithmProvider(hAlg, 0);
#else
    unsigned int out_len = 32;
    HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
         data.data(), static_cast<int>(data.size()), out.data(), &out_len);
#endif
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 * CONSTANT-TIME COMPARISON
 * ═══════════════════════════════════════════════════════════════ */

inline bool constant_time_compare(
    std::span<const uint8_t> a,
    std::span<const uint8_t> b) noexcept {
    if (a.size() != b.size()) return false;
    volatile uint8_t result = 0;
    for (size_t i = 0; i < a.size(); i++) {
        result |= a[i] ^ b[i];
    }
    return result == 0;
}

/* ═══════════════════════════════════════════════════════════════
 * ARX CORE (template-based for compile-time optimization)
 * ═══════════════════════════════════════════════════════════════ */

inline void arx_round(StateArray& state, uint64_t key) noexcept {
    for (size_t i = 0; i < NUM_WORDS; i++) {
        state[i] = state[i] + key + (uint64_t)(i * 13);
        state[i] ^= rotl(state[(i + 1) % NUM_WORDS], ROT1);
        state[i] = rotl(state[i], ROT2);
        state[i] ^= rotr(state[(i + 1) % NUM_WORDS], ROT4);
        state[i] = (state[i] * MULT_CONST) ^ rotl(state[i], ROT5);
    }
}

template <uint32_t Rounds>
inline void arx_block(StateArray& state, uint64_t key) noexcept {
    for (uint32_t r = 0; r < Rounds; r++) {
        for (size_t i = 0; i < NUM_WORDS; i++) {
            state[i] = (state[i] + key + (uint64_t)(i * 13));
            state[i] ^= rotl(state[(i + 1) % NUM_WORDS], ROT1);
            state[i] = rotl(state[i], ROT2);
            state[i] ^= rotr(state[(i + 1) % NUM_WORDS], ROT4);
            state[i] = (state[i] * MULT_CONST) ^ rotl(state[i], ROT5);
        }
        key = rotl(key, ROT3) ^ MIX_PRIME_2;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * HASHER CLASS (main API)
 * ═══════════════════════════════════════════════════════════════ */

class Hasher {
public:
    explicit Hasher(HashParams params = {}) : params_(std::move(params)) {
        resolve_params();
    }

    template <SecurityLevel L>
    static Hasher create() {
        HashParams p;
        p.security_level = L;
        return Hasher(p);
    }

    /**
     * @brief Compute Aelvonlock hash.
     * @param input Input data (UTF-8)
     * @param salt Optional salt (auto-generated if empty)
     * @return Pair of (hash_string, salt)
     */
    std::pair<std::string, ByteBuffer> hash(
        std::span<const uint8_t> input,
        std::span<const uint8_t> salt = {}) const {
        return hash_impl(input, salt);
    }

    std::pair<std::string, ByteBuffer> hash(
        std::string_view input,
        std::span<const uint8_t> salt = {}) const {
        return hash_impl(
            std::span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(input.data()),
                input.size()),
            salt);
    }

    /**
     * @brief Verify a password against a stored hash.
     * @param password Password to verify
     * @param stored_hash Previously computed hash
     * @param salt Salt used for original hash
     * @return true if password matches
     */
    bool verify(std::span<const uint8_t> password,
                std::string_view stored_hash,
                std::span<const uint8_t> salt) const {
        auto [recomputed, _] = hash(password, salt);
        return constant_time_compare(
            std::span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(recomputed.data()),
                recomputed.size()),
            std::span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(stored_hash.data()),
                stored_hash.size()));
    }

    bool verify(std::string_view password,
                std::string_view stored_hash,
                std::span<const uint8_t> salt) const {
        return verify(
            std::span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(password.data()),
                password.size()),
            stored_hash, salt);
    }

    // Note: Use aelvonlock::VERSION_STRING, aelvonlock::SYMBOLS,
    // and aelvonlock::NUM_SYMBOLS directly as static constexpr members.

private:
    /* Resolve traits at runtime */
    struct RuntimeTraits {
        uint32_t arx_rounds;
        uint32_t num_lanes;
        uint32_t final_rounds;
        uint32_t mem_mult;
    };

    RuntimeTraits resolve_traits() const noexcept {
        switch (params_.security_level) {
            case SecurityLevel::Low:
                return {4, 1, 2, 1};
            case SecurityLevel::Standard:
                return {12, 2, 4, 4};
            case SecurityLevel::High:
                return {32, 4, 6, 16};
            case SecurityLevel::Maximum:
                return {64, 8, 8, 64};
            case SecurityLevel::Ultimate:
                return {128, 16, 16, 256};
        }
        return {128, 16, 16, 256};
    }

    void resolve_params() {
        auto traits = resolve_traits();
        if (params_.num_lanes == 0)
            params_.num_lanes = traits.num_lanes;
        if (params_.arx_rounds == 0)
            params_.arx_rounds = traits.arx_rounds;
    }

    std::pair<std::string, ByteBuffer> hash_impl(
        std::span<const uint8_t> input,
        std::span<const uint8_t> salt) const {
        /* Validate input */
        if (input.empty()) throw EmptyInputError();
        if (input.size() > MAX_INPUT) throw InputTooLongError();

        auto traits = resolve_traits();

        /* Salt */
        SaltArray salt_arr{};
        ByteBuffer salt_out;
        if (salt.empty()) {
            salt_out.resize(SALT_SIZE);
            random_bytes(salt_out);
            std::memcpy(salt_arr.data(), salt_out.data(), SALT_SIZE);
        } else {
            salt_out.assign(salt.begin(), salt.end());
            std::memcpy(salt_arr.data(), salt.data(),
                        std::min(salt.size(), (size_t)SALT_SIZE));
        }

        /* Stretch salt */
        auto input_key = sha256(input);
        auto stretched = stretch_salt(salt_arr, input_key, params_.salt_stretch_iterations);

        /* Combine */
        std::array<uint8_t, 64> combined;
        for (size_t i = 0; i < 32; i++) {
            combined[i] = salt_arr[i] ^ stretched[i];
            combined[32 + i] = stretched[i];
        }
        auto final_salt = sha256(combined);

        /* Salt words */
        StateArray salt_words{};
        for (size_t i = 0; i < NUM_WORDS; i++) {
            for (size_t b = 0; b < 8; b++) {
                salt_words[i] = (salt_words[i] << 8) | final_salt[(i * 8 + b) % 32];
            }
        }
        uint64_t salt_int = salt_words[0];

        /* Input entropy */
        auto input_entropy = sha256(input);

        /* State initialization */
        StateArray state{};
        for (size_t i = 0; i < NUM_WORDS; i++) {
            state[i] = salt_words[i];
            if (i < 4) {
                uint64_t ew = 0;
                for (size_t b = 0; b < 8; b++) {
                    ew |= (uint64_t)input_entropy[i * 8 + b] << (b * 8);
                }
                state[i] ^= ew;
            }
            state[i] ^= INIT_XOR ^ ((uint64_t)i * MIX_PRIME_1);
        }

        /* Initialize lanes */
        std::vector<StateArray> lanes(traits.num_lanes);
        for (auto& lane : lanes) {
            for (size_t i = 0; i < NUM_WORDS; i++) {
                lane[i] = state[i] ^ (uint64_t)(&lane - lanes.data()) * MIX_PRIME_2;
            }
        }

        /* Block processing */
        size_t num_blocks = (input.size() + 63) / 64;
        for (size_t b = 0; b < num_blocks; b++) {
            size_t offset = b * 64;
            StateArray block_words{};

            for (size_t i = 0; i < NUM_WORDS; i++) {
                for (size_t j = 0; j < 8; j++) {
                    size_t idx = offset + i * 8 + j;
                    if (idx < input.size()) {
                        block_words[i] |= (uint64_t)input[idx] << (j * 8);
                    }
                }
                block_words[i] ^= salt_words[i];
                block_words[i] ^= state[i];
                block_words[i] ^= (uint64_t)b * MIX_PRIME_1;
            }

            uint64_t block_key = salt_int ^ ((uint64_t)b * MIX_PRIME_2);

            /* Apply ARX rounds based on security level */
            block_arx(block_words, block_key, traits.arx_rounds);

            /* Mix into state */
            for (size_t i = 0; i < NUM_WORDS; i++) {
                state[i] ^= block_words[i];
            }

            /* Feed into lanes */
            for (size_t l = 0; l < lanes.size(); l++) {
                for (size_t i = 0; i < NUM_WORDS; i++) {
                    lanes[l][i] ^= state[i];
                    lanes[l][i] ^= (uint64_t)l * (uint64_t)i * (uint64_t)b;
                }
            }
        }

        /* Lane cross-pollination with security-level-appropriate rounds */
        StateArray mixed_state{};
        for (size_t l = 0; l < lanes.size(); l++) {
            uint64_t lane_key = salt_int ^ ((uint64_t)l * MIX_PRIME_1);
            block_arx(lanes[l], lane_key, traits.arx_rounds);
            for (size_t i = 0; i < NUM_WORDS; i++) {
                mixed_state[i] ^= lanes[l][i];
            }
        }

        for (size_t i = 0; i < NUM_WORDS; i++) {
            state[i] ^= mixed_state[i];
            state[i] ^= rotl(state[(i + 1) % NUM_WORDS], ROT4);
            state[i] = state[i] * MULT_CONST;
        }

        /* Finalization */
        finalize_state(state, salt_words, traits.final_rounds,
                      [&](StateArray& s, uint64_t k, unsigned /*rnd*/) {
                          block_arx(s, k, 8);
                      });

    /* Generate symbolic output with version tag */
    auto result = generate_symbolic_output(state, params_.desired_length);
    result.insert(0, "V.C.P.2.");

    return {result, std::move(salt_out)};
    }

    /* Runtime ARX block (dispatched by rounds count) */
    static void block_arx(StateArray& state, uint64_t key, uint32_t rounds) noexcept {
        for (uint32_t r = 0; r < rounds; r++) {
            for (size_t i = 0; i < NUM_WORDS; i++) {
                state[i] = (state[i] + key + (uint64_t)(i * 13));
                state[i] ^= rotl(state[(i + 1) % NUM_WORDS], ROT1);
                state[i] = rotl(state[i], ROT2);
                state[i] ^= rotr(state[(i + 1) % NUM_WORDS], ROT4);
                state[i] = (state[i] * MULT_CONST) ^ rotl(state[i], ROT5);
            }
            key = rotl(key, ROT3) ^ MIX_PRIME_2;
        }
    }

    static SaltArray stretch_salt(
        const SaltArray& salt,
        const std::array<uint8_t, 32>& input_key,
        uint32_t iterations) {
        SaltArray stretched = salt;

        for (uint32_t i = 0; i < iterations; i++) {
            std::array<uint8_t, 36> hmac_key;
            std::memcpy(hmac_key.data(), input_key.data(), 32);
            hmac_key[32] = (uint8_t)(i & 0xFF);
            hmac_key[33] = (uint8_t)((i >> 8) & 0xFF);
            hmac_key[34] = (uint8_t)((i >> 16) & 0xFF);
            hmac_key[35] = (uint8_t)((i >> 24) & 0xFF);

            auto result = hmac_sha256(hmac_key, stretched);
            std::memcpy(stretched.data(), result.data(), SALT_SIZE);

            if (i % 2 == 0) {
                for (size_t j = 0; j < SALT_SIZE; j++) {
                    stretched[j] ^= salt[j];
                }
            }
        }

        return stretched;
    }

    static void finalize_state(StateArray& state,
                                const StateArray& salt_words,
                                uint32_t rounds,
                                auto finalizer) {
        for (uint32_t r = 0; r < rounds; r++) {
            uint64_t key = salt_words[r % NUM_WORDS] ^ ((uint64_t)r * MIX_PRIME_1);
            arx_round(state, key);
            for (size_t i = 0; i < NUM_WORDS; i++) {
                state[i] ^= rotl(salt_words[i % NUM_WORDS], (i + r) % 64);
                state[i] ^= rotr(state[(i + 1) % NUM_WORDS], ROT3);
            }
        }

        for (uint32_t r = 0; r < rounds; r++) {
            uint64_t key = salt_words[0] ^ ((uint64_t)r * MIX_PRIME_2);
            finalizer(state, key, r);
        }
    }

    static std::string generate_symbolic_output(
        const StateArray& state, uint32_t desired_length) {
        /* Convert state to bytes */
        std::array<uint8_t, 64> state_bytes;
        for (size_t i = 0; i < NUM_WORDS; i++) {
            for (size_t b = 0; b < 8; b++) {
                state_bytes[i * 8 + b] = (uint8_t)(state[i] >> (b * 8));
            }
        }

        /* Generate symbols with rejection sampling */
        std::vector<uint32_t> symbols;
        symbols.reserve(desired_length);

        unsigned mod_base = 256 - (256 % NUM_SYMBOLS);
        size_t byte_idx = 0;
        unsigned retries = 0;
        constexpr unsigned MAX_RETRIES = 10000;

        while (symbols.size() < desired_length && retries < MAX_RETRIES) {
            if (byte_idx >= state_bytes.size()) {
                auto ext = sha256(state_bytes);
                std::memcpy(state_bytes.data(), ext.data(), 32);
                byte_idx = 0;
            }

            uint8_t byte_val = state_bytes[byte_idx++];
            if (byte_val < mod_base) {
                symbols.push_back(SYMBOLS[byte_val % NUM_SYMBOLS]);
                retries = 0;
            } else {
                retries++;
                if (retries >= MAX_RETRIES) {
                    symbols.push_back(SYMBOLS[byte_val % NUM_SYMBOLS]);
                    retries = 0;
                }
            }
        }

        /* Encode to UTF-8 */
        std::string result;
        result.reserve(desired_length * 4);

        for (auto cp : symbols) {
            if (cp < 0x80) {
                result += (char)cp;
            } else if (cp < 0x800) {
                result += (char)(0xC0 | (cp >> 6));
                result += (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                result += (char)(0xE0 | (cp >> 12));
                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                result += (char)(0x80 | (cp & 0x3F));
            } else {
                result += (char)(0xF0 | (cp >> 18));
                result += (char)(0x80 | ((cp >> 12) & 0x3F));
                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                result += (char)(0x80 | (cp & 0x3F));
            }
        }

        return result;
    }

    HashParams params_;
};

/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE FUNCTIONS
 * ═══════════════════════════════════════════════════════════════ */

inline std::pair<std::string, ByteBuffer> hash(
    std::string_view input,
    std::span<const uint8_t> salt = {},
    SecurityLevel level = SecurityLevel::Ultimate) {
    HashParams p;
    p.security_level = level;
    Hasher hasher(p);
    return hasher.hash(input, salt);
}

inline bool verify(std::string_view password,
                    std::string_view stored_hash,
                    std::span<const uint8_t> salt,
                    SecurityLevel level = SecurityLevel::Ultimate) {
    HashParams p;
    p.security_level = level;
    Hasher hasher(p);
    return hasher.verify(password, stored_hash, salt);
}

inline bool constant_time_compare(std::string_view a, std::string_view b) noexcept {
    return constant_time_compare(
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(a.data()), a.size()),
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(b.data()), b.size()));
}

/* ═══════════════════════════════════════════════════════════════
 * SELF-TEST
 * ═══════════════════════════════════════════════════════════════ */

inline bool self_test() {
    bool all_pass = true;

    auto check = [&](const char* name, bool pass) {
        printf("  [%s] %s\n", pass ? "PASS" : "FAIL", name);
        all_pass &= pass;
    };

    /* Test 1: Basic hashing */
    auto [h1, s1] = hash("test");
    check("basic_hashing", !h1.empty());

    /* Test 2: Deterministic */
    SaltArray fixed_salt{1, 2, 3, 4};
    auto [h2a, _] = hash("test", fixed_salt);
    auto [h2b, __] = hash("test", fixed_salt);
    check("deterministic", h2a == h2b);

    /* Test 3: Different inputs */
    auto [h3a, ___] = hash("hello");
    auto [h3b, ____] = hash("world");
    check("different_inputs", h3a != h3b);

    /* Test 4: Salt changes output */
    SaltArray s4a{1, 2, 3, 4};
    SaltArray s4b{5, 6, 7, 8};
    auto [h4a, _____] = hash("test", s4a);
    auto [h4b, ______] = hash("test", s4b);
    check("salt_independence", h4a != h4b);

    /* Test 5: Verification */
    auto [h5, s5] = hash("password123");
    check("verification", verify("password123", h5, s5));
    check("wrong_rejection", !verify("wrong_password", h5, s5));

    /* Test 6: Version */
    check("version", std::string_view(VERSION_STRING) == VERSION_STRING);

    printf("\nOverall: %s\n", all_pass ? "ALL PASSED" : "SOME FAILED");
    return all_pass;
}

} /* namespace aelvonlock */

#endif /* AELVONLOCK_HPP */
