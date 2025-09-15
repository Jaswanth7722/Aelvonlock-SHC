# Aelvonlock - SHC (Symbolic Hashing Cryptography)

---

## What is Aelvonlock?

Aelvonlock is a Symbolic Hashing Cryptography (SHC) algorithm and the first implementation of this new category. Unlike traditional hashing systems that rely on encodings such as UTF-8, Base64, or ASCII, Aelvonlock operates entirely on unique symbolic representations. It introduces a set of 94 symbols not found in any existing encoding standard, ensuring that both the hashing process and the final output remain non-traditional. It contains a symbol map that consists of unicodes.

Aelvonlock design combines symbolic obfuscation with memory-hard cryptographic principles, making it resistant to brute-force attempts. Aelvonlock produces a 512-bit output, which offers strong resistance against collision and birthday attacks.

Aelvonlock is a versatile cryptography framework that provides highly secure hashing while maintaining flexibility for different use cases. Its design emphasizes:

- **Symbolic hashing**: Unique symbolic output representation for enhanced obscurity.  
- **Memory-hard operation**: Configurable memory consumption for defense against modern hardware attacks.  
- **Enterprise readiness**: Optimized for integration in API services, desktop applications, and cloud deployments.  

Unlike traditional hashing algorithms, Aelvonlock focuses on both **security and conceptual clarity**, enabling developers and researchers to implement advanced security measures without exposing sensitive internals.

---

## Features

- **Symbolic Hash Output**: Transforms input into an abstract symbolic representation, preventing direct pattern recognition.  
- **Memory-Hard Modes**: Configurable memory usage to resist GPU/ASIC attacks.  
- **Multiple Variants**: Optimized modes for speed, security, or balance.  
- **Salt Support**: Secure, unique salts per hash to prevent rainbow table attacks.  
- **Cross-Platform Compatibility**: Works on Windows, Linux, and macOS.  
- **High Configurability**: Tailor hashing cost, memory usage, and symbolic complexity.

---

## Variants In Aelvonlock

Aelvonlock provides multiple variants for different use cases, with configurable memory-hardness levels (64MB, 128MB, 256MB, etc.) and security settings. Each variant comes with different security levels and execution times.

| Variant     | Description | Use Case |
|------------|-------------|----------|
| Maxlock    | Maximum memory and processing for top-level security | Enterprise systems handling highly sensitive data |
| Hardened   | High security with balanced memory usage | Servers, cloud APIs, and secure storage |
| Balanced   | Optimized for both performance and security | General-purpose applications |
| Lite/Mini  | Low memory and fast hashing | Embedded systems, low-resource devices |

Each variant allows fine-grained control of hashing parameters to fit deployment constraints.

---

## Architecture Overview

Aelvonlock operates through a modular, layered workflow:

1. **Input Processing**: Accepts passwords or data inputs with optional salts.  
2. **Memory-Hard Computation**: Configurable memory matrix used to resist hardware-based attacks.  
3. **Symbolic Encoding**: Abstract symbolic transformation of the output.  
4. **Finalization & Verification**: Generates a unique hash; provides a secure verification mechanism without revealing internal states.

> Note: The internal algorithmic steps, ARX rounds, and obfuscation techniques are proprietary and intentionally abstracted to maintain security.

---

## Installation

### Prerequisites

- Python ≥ 3.10  
- Optional: NumPy and Numba for optimized performance

### Steps

```bash
# Clone the repository
git clone https://github.com/jaswanth7722/aelvonlock.git
cd aelvonlock

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: test installation
python -c "from aelvonlock import Aelvonlock; print('Aelvonlock imported successfully')"
