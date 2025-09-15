# Aelvonlock-512 Variants Security Overview

This document compares all **Aelvonlock-512 hashing variants** with a focus on **security, memory usage, and cryptographic strength**.

---

## 1\. Maxlock

* **Memory Usage:** ~1 GB (very high)
* **ARX Rounds:** 32 + 128 main iterations, Finalize 5 rounds
* **Salt Complexity:** Very high (per-byte mutation, randomized)
* **Lane Mixing:** Full multi-lane (4+ parallel lanes merged)
* **Avalanche Effect:** Excellent (high diffusion across all bits)
* **Attack Resistance:**

  * Resistant to brute-force attacks
  * High resistance to collision attacks
  * Strong against memory–time tradeoff attacks

* **Use Case:** Critical security applications where maximum protection is required, tolerates high CPU/memory usage

---

## 2\. Hardened

* **Memory Usage:** 128–280 MB (configurable)
* **ARX Rounds:** Medium to high
* **Salt Complexity:** Moderate to high
* **Lane Mixing:** Partial multi-lane
* **Avalanche Effect:** Good (strong bit diffusion, better than Balanced)
* **Attack Resistance:**

  * Resistant to standard brute-force attacks
  * Good collision resistance
  * Moderate resistance to memory attacks

* **Use Case:** High-security scenarios needing strong hashing without extreme resource usage

---

## 3\. Balanced

* **Memory Usage:** ~128 MB
* **ARX Rounds:** Moderate (8–16 rounds)
* **Salt Complexity:** Moderate
* **Lane Mixing:** Minimal
* **Avalanche Effect:** Moderate
* **Attack Resistance:**

  * Acceptable resistance against brute-force
  * Standard collision resistance
  * Basic memory-hard protection

* **Use Case:** Default choice for applications where speed and security need a balanced tradeoff

---

## 4\. Lite

* **Memory Usage:** ~64 MB
* **ARX Rounds:** 4
* **Salt Complexity:** Basic (simple XOR)
* **Lane Mixing:** None
* **Avalanche Effect:** Weak
* **Attack Resistance:**

  * Low resistance to brute-force
  * Weak collision protection
  * Minimal memory-hard defense

* **Use Case:** Fast hashing, benchmarking, or low-security scenarios

---

## 5\. Mini (64MB)

* **Memory Usage:** ~64 MB
* **ARX Rounds:** 4
* **Salt Complexity:** Basic
* **Lane Mixing:** None
* **Avalanche Effect:** Weak
* **Attack Resistance:** Same as Lite
* **Use Case:** Ultra-lightweight hashing, embedded or constrained devices

---

**Key Takeaways:**

* **Maxlock:** Maximum cryptographic security at the cost of memory/CPU.
* **Hardened:** High security, moderate resource use.
* **Balanced:** Good compromise for general-purpose applications.
* **Lite \& Mini:** Fastest variants, weak security—use only for low-risk scenarios.
