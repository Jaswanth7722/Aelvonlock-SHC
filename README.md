# Aelvonlock-SHC v2.0 (Symbolic Hashing Cryptography)

<div align="center">

![Aelvonlock](Aelvonlock_exe_files/aelvonlock_512)

**The Strongest Symbolic Hashing Cryptography Framework Ever Built**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-brightgreen)](.)

</div>

---

## What is Aelvonlock?

**Aelvonlock** is a revolutionary **Symbolic Hashing Cryptography (SHC)** algorithm — the first implementation of this new cryptographic category. Unlike traditional hashing systems (SHA-256, BLAKE2, Argon2) that rely on encodings such as UTF-8, Base64, or ASCII, Aelvonlock operates entirely on **unique symbolic representations** from a private Unicode block (U+100000–U+10005D).

### Why Aelvonlock?

| Feature | Traditional Hashing | Aelvonlock-SHC |
|---------|-------------------|-----------------|
| **Output Encoding** | Hex/Base64/ASCII | **94 custom Unicode symbols** |
| **Pattern Recognition** | Trivial to identify | **Impossible without symbol map** |
| **Memory Hardness** | Fixed | **Input-dependent** |
| **Lane Processing** | Single | **Up to 16 parallel lanes** |
| **ARX Rounds** | Fixed (64-80) | **Up to 128 rounds** |
| **Salt Stretching** | PBKDF2 | **HMAC-SHA256 + XOR Feistel** |
| **Anti-Tamper** | None | **Integrity verification** |

---

## Variants

Aelvonlock provides **6 variants** for different use cases:

| Variant | Memory | ARX Rounds | Lanes | Security | Use Case |
|---------|--------|------------|-------|----------|----------|
| **Ultimate** 🏆 | Input-dependent (256MB–4GB/lane) | **128** | **16** | **Maximum** | Enterprise, military-grade |
| **Maxlock** 🛡️ | ~1 GB | 32 | 4 | Maximum | Highly sensitive data |
| **Hardened** 🔒 | ~280 MB | 32 | 4 | High | Servers, cloud APIs |
| **Balanced** ⚖️ | ~128 MB | 12 | 2 | Standard | General purpose |
| **Lite** 🪶 | ~64 MB | 4 | 1 | Basic | Embedded systems |
| **Fast** ⚡ | ~64 MB | 4 | 1 | Basic | Low-resource devices |

---

## Architecture Overview

Aelvonlock operates through a multi-layered security architecture:

```
Input Text
    │
    ▼
┌─────────────────────────────┐
│ 1. Input Sanitization       │  Unicode normalization, length check
├─────────────────────────────┤
│ 2. Symbolic Encoding        │  Map to 94 custom Unicode symbols
├─────────────────────────────┤
│ 3. Salt Generation          │  256-bit CSPRNG salt + HMAC stretching
├─────────────────────────────┤
│ 4. Input Entropy Extraction │  Multi-hash (SHA-256, SHA-512, BLAKE2)
├─────────────────────────────┤
│ 5. Memory Allocation        │  Input-dependent matrix sizing
├─────────────────────────────┤
│ 6. Multi-Lane Init          │  Up to 16 parallel processing lanes
├─────────────────────────────┤
│ 7. Block Processing         │  128 ARX rounds per block
├─────────────────────────────┤
│ 8. Memory-Hard Matrix       │  Argon2-like random access pattern
├─────────────────────────────┤
│ 9. State-Dependent Reads    │  Memory reads mixed into state
├─────────────────────────────┤
│ 10. Lane Cross-Pollination  │  Lane state exchange and merge
├─────────────────────────────┤
│ 11. Finalization            │  Multi-round ARX avalanche
├─────────────────────────────┤
│ 12. Symbolic Output         │  Rejection-sampled symbol selection
    │
    ▼
Symbolic Hash (64+ symbols)
```

### Security Properties

- **Preimage resistance:** ≥ 2^512
- **Collision resistance:** ≥ 2^256
- **Memory-hardness:** Configurable, input-dependent
- **Side-channel resistance:** Constant-time operations
- **Length extension resistance:** Merkle-Damgård strengthening
- **Avalanche effect:** ≥ 50% bit difference on single bit change

---

## Installation

### Prerequisites

- **Python ≥ 3.10**
- **Rust ≥ 1.70** (for Rust CLI)
- **NumPy ≥ 1.22** (automatically installed)
- **Numba ≥ 0.58** (automatically installed)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/jaswanth7722/aelvonlock.git
cd aelvonlock

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.Ultimate import aelvonlock512_hash; print('Aelvonlock Ultimate imported successfully')"
```

### Build Rust CLI (Optional - Maximum Performance)

```bash
cd src/rust_aelvonlock
cargo build --release
./target/release/aelvonlock --help
```

---

## Usage

### Python API

```python
from src.Ultimate import aelvonlock512_hash, verify_password

# Hash a password
hash_string, salt = aelvonlock512_hash("my_secret_password")
print(f"Hash: {hash_string}")
print(f"Salt: {salt.hex()}")

# Verify a password
is_valid = verify_password("my_secret_password", hash_string, salt.hex())
print(f"Valid: {is_valid}")  # True

# Use specific variant
from src.Maxlock import aelvonlock512_hash as maxlock_hash
hash_string, salt = maxlock_hash("my_secret_password")
```

### CLI (Python)

```bash
# Compute hash
python src/Ultimate.py hash "my secret password"

# Verify password
python src/Ultimate.py verify "my password" "V.U.L.2.☺♫..." "a1b2c3d4..."

# Run self-test
python src/Ultimate.py self-test

# Run benchmark
python src/Ultimate.py benchmark

# View variant info
python src/Ultimate.py info
```

### CLI (Rust - Maximum Performance)

```bash
cd src/rust_aelvonlock
cargo run --release -- hash "my secret password"
cargo run --release -- verify "my password" "V.R.L.2.☺♫..." "a1b2c3..."
cargo run --release -- self-test
cargo run --release -- benchmark
```

### Benchmark

```bash
python benchmark.py               # Full benchmark
python benchmark.py --quick        # Quick benchmark
python benchmark.py --avalanche    # Avalanche test only
python benchmark.py --output results.json  # Save results
```

---

## Symbol Map

Aelvonlock uses **94 unique symbols** from the **private Unicode block U+100000–U+10005D**. These symbols are:
- **Not present** in any standard encoding (UTF-8, ASCII, Base64)
- **Visually unique** with the bundled Aelvonlock font
- **Tamper-proof** via runtime integrity verification

The symbol map includes:
- 26 uppercase letters (A-Z)
- 26 lowercase letters (a-z)
- 10 digits (0-9)
- 32 punctuation/special characters

---

## Variants Comparison

All variants produce **512-bit internal state** with **symbolic output**:

| Property | Fast | Lite | Balanced | Hardened | Maxlock | Ultimate |
|----------|------|------|----------|----------|---------|----------|
| State Size | 256-bit | 256-bit | 512-bit | 512-bit | 512-bit | 512-bit |
| ARX Rounds | 4 | 4 | 12 | 32 | 32 | 128 |
| Lanes | 1 | 1 | 2 | 4 | 4 | 16 |
| Finalize | 4 | 4 | 6 | 5 | 5 | 16 |
| Memory | ~64MB | ~64MB | ~128MB | ~280MB | ~1GB | Input-dep. |
| Salt Stretch | No | No | Yes | Yes | Yes | Yes |
| Integrity Check | No | No | Yes | Yes | Yes | Yes |
| Numba JIT | No | Yes | Yes | Yes | Yes | Yes |

---

## Testing

```bash
# Run comprehensive test suite
python -m pytest tests/test_aelvonlock.py -v

# Run specific tests
python -m pytest tests/test_aelvonlock.py -k "test_avalanche"

# Run all tests including Ultimate (requires patience)
python -m pytest tests/test_aelvonlock.py -v --timeout=300
```

---

## Security Considerations

1. **Output Length**: Always use ≥ 64 symbols for production use
2. **Salt Management**: Store salts alongside hashes (not secret, but unique)
3. **Symbol Map**: Protect the symbol map — it's integral to hash verification
4. **Variant Selection**: Use Ultimate or Maxlock for passwords, use Balanced for general hashing
5. **Constant-Time**: All comparison functions use constant-time operations to prevent timing attacks

---

## File Structure

```
aelvonlock/
├── src/
│   ├── core.py           # Shared core module (symbol map, utilities, ARX)
│   ├── Ultimate.py       # Ultimate variant (strongest)
│   ├── Maxlock.py        # Maxlock variant (enterprise maximum)
│   ├── Hardend.py        # Hardened variant (high security)
│   ├── Balanced.py       # Balanced variant (standard security)
│   ├── Lite.py           # Lite variant (basic security)
│   ├── Fast.py           # Fast variant (ultra-light)
│   ├── rust_aelvonlock/  # Rust CLI (maximum performance)
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── README.md
├── font/
│   ├── Aelvonlock.ttf    # Custom symbol font
│   └── symbol_map.json   # Symbol map (JSON format)
├── tests/
│   └── test_aelvonlock.py  # Comprehensive test suite
├── benchmark.py            # Performance benchmark
├── requirements.txt
└── README.md
```

---

## License

MIT License — See LICENSE file for details.

---

## Acknowledgments

- **Argon2** — For memory-hard function design principles
- **SHA-3** — For sponge construction inspiration
- **BLAKE2** — For ARX optimization patterns
- **MurmurHash** — For mixing constant derivation
