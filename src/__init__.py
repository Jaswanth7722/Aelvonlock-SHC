"""
Aelvonlock-SHC (Symbolic Hashing Cryptography)
==============================================
The strongest symbolic hashing cryptography framework.

Variants:
- Ultimate: Maximum security with input-dependent memory
- Maxlock: Enterprise maximum security (~1 GB memory)
- Hardened: High security for servers (~280 MB memory)
- Balanced: Standard security (~128 MB memory)
- Lite: Basic security (~64 MB memory)
- Fast: Ultra-fast for embedded devices (~64 MB memory)
"""

from src.core import (
    VERSION,
    VERSION_NAME,
    SYMBOL_MAP,
    SYMBOL_LIST,
    sanitize_input,
)

# Version info
__version__ = VERSION
__author__ = "Aelvonlock Developers"
__description__ = "Symbolic Hashing Cryptography Framework"

# Available variants
VARIANTS = [
    "Ultimate",
    "Maxlock",
    "Hardened",
    "Balanced",
    "Lite",
    "Fast",
]
