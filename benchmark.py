#!/usr/bin/env python
"""
Aelvonlock-SHC Benchmark Suite
===============================
Measures and compares performance across all variants.

Usage:
    python benchmark.py              # Full benchmark
    python benchmark.py --quick      # Quick benchmark (1 iteration)
    python benchmark.py --output report.json  # Save results
"""

import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import core

# Try importing all variants
VARIANT_MODULES: Dict[str, Any] = {}

try:
    from src import Fast as _fast
    VARIANT_MODULES["Fast"] = _fast
except ImportError:
    pass

try:
    from src import Lite as _lite
    VARIANT_MODULES["Lite"] = _lite
except ImportError:
    pass

try:
    from src import Balanced as _balanced
    VARIANT_MODULES["Balanced"] = _balanced
except ImportError:
    pass

try:
    from src import Hardend as _hardened
    VARIANT_MODULES["Hardened"] = _hardened
except ImportError:
    pass

try:
    from src import Maxlock as _maxlock
    VARIANT_MODULES["Maxlock"] = _maxlock
except ImportError:
    pass

try:
    from src import Ultimate as _ultimate
    VARIANT_MODULES["Ultimate"] = _ultimate
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# BENCHMARK UTILITIES
# ═══════════════════════════════════════════════════════════════

def time_hash(func: Callable, text: str, iterations: int = 3) -> float:
    """Time a single hash function over multiple iterations."""
    # Warmup
    try:
        func(text)
    except Exception:
        pass

    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func(text)
        except Exception:
            return float('inf')
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def get_hash_function(module, variant_name: str) -> Optional[Callable]:
    """Get the hash function from a variant module."""
    if hasattr(module, 'aelvonlock512_hash'):
        return module.aelvonlock512_hash
    return None


def run_benchmark(text: str = "benchmark_test_input",
                  iterations: int = 3,
                  quick: bool = False) -> Dict[str, Any]:
    """Run benchmark across all available variants."""
    if quick:
        iterations = 1

    results: Dict[str, Any] = {
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "input": {
            "text": text,
            "length": len(text),
            "iterations": iterations,
            "quick": quick,
        },
        "results": {},
    }

    print(f"\n{'='*60}")
    print(f"Aelvonlock-SHC Benchmark Suite")
    print(f"{'='*60}")
    print(f"Input: '{text}' ({len(text)} chars)")
    print(f"Iterations per variant: {iterations}")
    print(f"{'='*60}\n")

    for variant_name, module in sorted(VARIANT_MODULES.items()):
        hash_func = get_hash_function(module, variant_name)
        if hash_func is None:
            continue

        print(f"  {variant_name}...", end=" ", flush=True)
        avg_time = time_hash(hash_func, text, iterations)

        if avg_time == float('inf'):
            print("[FAILED]")
            continue

        hashes_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')

        print(f"[{avg_time:.3f}s, {hashes_per_sec:.2f} h/s]")

        results["results"][variant_name] = {
            "avg_time_seconds": round(avg_time, 4),
            "hashes_per_second": round(hashes_per_sec, 2),
            "iterations": iterations,
        }

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")

    if results["results"]:
        sorted_results = sorted(
            results["results"].items(),
            key=lambda x: x[1]["hashes_per_second"],
            reverse=True,
        )
        print(f"{'Variant':<20} {'Time (s)':<15} {'Hashes/sec':<15}")
        print("-" * 50)
        for name, data in sorted_results:
            print(f"{name:<20} {data['avg_time_seconds']:<15.4f} {data['hashes_per_second']:<15.2f}")
    else:
        print("No variants available to benchmark!")

    return results


def run_avalanche_test(variant_name: str,
                       hash_func: Callable,
                       num_tests: int = 100) -> Dict[str, float]:
    """
    Test avalanche effect: measure how many output bits change
    when a single input bit is flipped.
    """
    total_diff_bits = 0
    total_bits = 0

    for i in range(num_tests):
        text1 = f"test_input_{i}"
        text2 = text1 + "x"  # One character change

        try:
            h1, _ = hash_func(text1)
            h2, _ = hash_func(text2)
        except Exception:
            continue

        # Convert hash strings to bytes for bit comparison
        min_len = min(len(h1), len(h2))
        for a, b in zip(h1[:min_len], h2[:min_len]):
            xor_val = ord(a) ^ ord(b)
            total_diff_bits += bin(xor_val).count('1')
            total_bits += 8

    if total_bits == 0:
        return {"avg_diff_bits": 0, "diff_percentage": 0}

    avg_diff = total_diff_bits / total_bits
    diff_pct = (total_diff_bits / total_bits) * 100

    return {
        "avg_diff_bits": round(avg_diff, 4),
        "diff_percentage": round(diff_pct, 2),
    }


def full_benchmark(quick: bool = False, output_file: Optional[str] = None):
    """Run complete benchmark with performance and avalanche testing."""
    results = run_benchmark("benchmark_test_input_aelvonlock", 1 if quick else 3, quick)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


def quick_comparison():
    """Quick performance comparison between all variants."""
    print("\nAelvonlock-SHC Quick Comparison")
    print("=" * 60)

    texts = [
        "short",
        "medium_length_input_test",
        "a" * 100,
    ]

    for text in texts:
        print(f"\nInput: '{text}' ({len(text)} chars)")
        times: List[Tuple[str, float]] = []

        for variant_name, module in sorted(VARIANT_MODULES.items()):
            hash_func = get_hash_function(module, variant_name)
            if hash_func is None:
                continue

            start = time.perf_counter()
            try:
                hash_func(text)
                elapsed = time.perf_counter() - start
                times.append((variant_name, elapsed))
                print(f"  {variant_name:<15} {elapsed:.4f}s")
            except Exception as e:
                print(f"  {variant_name:<15} [FAIL] {e}")

        if times:
            fastest = min(times, key=lambda x: x[1])
            slowest = max(times, key=lambda x: x[1])
            ratio = slowest[1] / fastest[1] if fastest[1] > 0 else 0
            print(f"  {'-'*40}")
            print(f"  Fastest: {fastest[0]} ({fastest[1]:.4f}s)")
            print(f"  Slowest: {slowest[0]} ({slowest[1]:.4f}s)")
            print(f"  Ratio: {ratio:.1f}x")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aelvonlock-SHC Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (1 iteration)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--avalanche", action="store_true", help="Run avalanche test only")
    parser.add_argument("--compare", action="store_true", help="Quick comparison across input sizes")

    args = parser.parse_args()

    if args.avalanche:
        print("\nAvalanche Effect Analysis")
        print("=" * 60)
        for variant_name, module in sorted(VARIANT_MODULES.items()):
            hash_func = get_hash_function(module, variant_name)
            if hash_func:
                aval = run_avalanche_test(variant_name, hash_func, 10)
                print(f"  {variant_name:<15} {aval['diff_percentage']:.1f}%")
    elif args.compare:
        quick_comparison()
    else:
        full_benchmark(args.quick, args.output)
