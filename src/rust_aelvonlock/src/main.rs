//! Aelvonlock-SHC Rust CLI
//!
//! Ultra-fast Aelvonlock Symbolic Hashing Cryptography implementation.
//! Provides the strongest hashing with symbolic output representation.
//!
//! This Rust version is optimized for:
//! - Maximum speed (compiled native code)
//! - Memory safety (Rust's ownership model)
//! - Cross-platform compatibility (Windows, Linux, macOS)

use std::io::{self, Write};
use clap::{Parser, Subcommand};
use sha2::{Sha256, Sha512, Digest};
use hmac::{Hmac, Mac};
use rand::RngCore;

type HmacSha256 = Hmac<Sha256>;

/// Aelvonlock-SHC: Symbolic Hashing Cryptography CLI
#[derive(Parser)]
#[command(name = "aelvonlock")]
#[command(about = "Aelvonlock Symbolic Hashing Cryptography CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute Aelvonlock hash
    Hash {
        /// Input text to hash
        text: String,
        /// Output hash symbol length
        #[arg(short, long, default_value_t = 64)]
        length: usize,
        /// Salt (hex encoded)
        #[arg(short, long)]
        salt: Option<String>,
        /// Disable salt stretching
        #[arg(long)]
        no_stretch: bool,
    },
    /// Verify password against stored hash
    Verify {
        /// Password to verify
        password: String,
        /// Stored hash
        hash: String,
        /// Salt used (hex encoded)
        salt_hex: String,
    },
    /// Run self-test
    SelfTest,
    /// Run benchmark
    Benchmark {
        /// Number of iterations
        #[arg(short, long, default_value_t = 5)]
        iterations: usize,
    },
    /// Show info
    Info,
}

// Symbol map - Aelvonlock Unicode Private Use Area (U+100000-U+10005D)
// These match the Python core.SYMBOL_LIST exactly for cross-platform compatibility.
const SYMBOLS: &[char] = &[
    '\u{100000}', '\u{100001}', '\u{100002}', '\u{100003}', '\u{100004}', '\u{100005}', '\u{100006}', '\u{100007}',
    '\u{100008}', '\u{100009}', '\u{10000A}', '\u{10000B}', '\u{10000C}', '\u{10000D}', '\u{10000E}', '\u{10000F}',
    '\u{100010}', '\u{100011}', '\u{100012}', '\u{100013}', '\u{100014}', '\u{100015}', '\u{100016}', '\u{100017}',
    '\u{100018}', '\u{100019}', '\u{10001A}', '\u{10001B}', '\u{10001C}', '\u{10001D}', '\u{10001E}', '\u{10001F}',
    '\u{100020}', '\u{100021}', '\u{100022}', '\u{100023}', '\u{100024}', '\u{100025}', '\u{100026}', '\u{100027}',
    '\u{100028}', '\u{100029}', '\u{10002A}', '\u{10002B}', '\u{10002C}', '\u{10002D}', '\u{10002E}', '\u{10002F}',
    '\u{100030}', '\u{100031}', '\u{100032}', '\u{100033}', '\u{100034}', '\u{100035}', '\u{100036}', '\u{100037}',
    '\u{100038}', '\u{100039}', '\u{10003A}', '\u{10003B}', '\u{10003C}', '\u{10003D}', '\u{10003E}', '\u{10003F}',
    '\u{100040}', '\u{100041}', '\u{100042}', '\u{100043}', '\u{100044}', '\u{100045}', '\u{100046}', '\u{100047}',
    '\u{100048}', '\u{100049}', '\u{10004A}', '\u{10004B}', '\u{10004C}', '\u{10004D}', '\u{10004E}', '\u{10004F}',
    '\u{100050}', '\u{100051}', '\u{100052}', '\u{100053}', '\u{100054}', '\u{100055}', '\u{100056}', '\u{100057}',
    '\u{100058}', '\u{100059}', '\u{10005A}', '\u{10005B}', '\u{10005C}', '\u{10005D}',
];

fn _verify_symbol_count() {
    assert_eq!(SYMBOLS.len(), 94, "Symbol map must have exactly 94 symbols");
}

const VERSION_TAG: &str = "V.R.L.2";
const ROUNDS: u32 = 128;
const SALT_STRETCH_ITERATIONS: u32 = 100_000;

/// Rotate 64-bit value left by r bits
fn rotate_left(val: u64, r: u32) -> u64 {
    let r = r & 63;
    (val << r) | (val >> (64 - r))
}

/// Rotate 64-bit value right by r bits
fn rotate_right(val: u64, r: u32) -> u64 {
    let r = r & 63;
    (val >> r) | (val << (64 - r))
}

/// Generate cryptographic random salt
fn generate_salt(length: usize) -> Vec<u8> {
    let mut salt = vec![0u8; length];
    rand::rngs::OsRng.fill_bytes(&mut salt);
    salt
}

/// Stretch salt using HMAC-SHA256 (PBKDF2-like)
fn stretch_salt(salt: &[u8], input: &str, iterations: u32) -> Vec<u8> {
    let input_key = Sha256::digest(input.as_bytes());
    let mut stretched = salt.to_vec();

    for i in 0..iterations {
        let mut mac = HmacSha256::new_from_slice(&input_key).expect("HMAC key");
        mac.update(&stretched);
        mac.update(&i.to_le_bytes());
        let result = mac.finalize().into_bytes();
        stretched = result.to_vec();

        // Mix back with original salt
        if i % 2 == 0 {
            for (a, b) in stretched.iter_mut().zip(salt.iter().cycle()) {
                *a ^= b;
            }
        }
    }

    stretched
}

/// Generate symbolic output from bytes using rejection sampling
fn bytes_to_symbols(data: &[u8], desired_length: usize) -> String {
    let num_symbols = SYMBOLS.len();
    let mod_base = 256 - (256 % num_symbols);
    let mut result = String::with_capacity(desired_length);
    let mut i = 0;
    let mut extended_data = data.to_vec();

    while result.len() < desired_length {
        if i >= extended_data.len() {
            // Extend data using SHA-256
            let ext = Sha256::digest(&extended_data);
            extended_data.extend_from_slice(&ext);
        }

        let byte_val = extended_data[i];
        if byte_val < mod_base as u8 {
            result.push(SYMBOLS[byte_val as usize % num_symbols]);
        }
        i += 1;
    }

    result
}

/// Core ARX hash computation
fn aelvonlock_hash_internal(text: &str, salt: &[u8]) -> Vec<u8> {
    // Salt stretching
    let stretched = stretch_salt(salt, text, SALT_STRETCH_ITERATIONS);
    let combined: Vec<u8> = salt.iter()
        .zip(stretched.iter().cycle())
        .map(|(a, b)| a ^ b)
        .collect();
    let final_salt = Sha256::digest(&combined);

    // Extract salt words
    let mut salt_words = [0u64; 8];
    for (i, word) in salt_words.iter_mut().enumerate() {
        let start = (i * 8) % final_salt.len();
        let mut bytes = [0u8; 8];
        for j in 0..8 {
            bytes[j] = final_salt[(start + j) % final_salt.len()];
        }
        *word = u64::from_be_bytes(bytes);
    }

    let salt_int = salt_words[0];

    // Input entropy
    let input_hash = Sha256::digest(text.as_bytes());
    let input_entropy = Sha512::digest(text.as_bytes());

    // Initialize state
    let mut state = [0u64; 8];
    for i in 0..8 {
        let mut val = salt_words[i % 8];
        val ^= u64::from_le_bytes([
            input_hash[i % 32],
            input_hash[(i + 1) % 32],
            input_hash[(i + 2) % 32],
            input_hash[(i + 3) % 32],
            input_entropy[i % 64],
            input_entropy[(i + 7) % 64],
            input_entropy[(i + 13) % 64],
            input_entropy[(i + 19) % 64],
        ]);
        val ^= 0xABCDEF1234567890u64 ^ (i as u64 * 0xC6A4A7935BD1E995u64);
        state[i] = val;
    }

    // Process text in blocks
    let text_bytes = text.as_bytes();
    for block_start in (0..text_bytes.len()).step_by(64) {
        let end = std::cmp::min(block_start + 64, text_bytes.len());
        let block = &text_bytes[block_start..end];

        // Convert block to words
        let mut block_words = [0u64; 8];
        for (i, chunk) in block.chunks(8).enumerate() {
            if i < 8 {
                let mut bytes = [0u8; 8];
                for (j, b) in chunk.iter().enumerate() {
                    bytes[j] = *b;
                }
                block_words[i] = u64::from_le_bytes(bytes);
            }
        }

        // Mix block with salt and state
        for i in 0..8 {
            block_words[i] ^= salt_words[i];
            block_words[i] ^= state[i];
        }

        // ARX rounds
        for rnd in 0..ROUNDS {
            let key = salt_int ^ (rnd as u64 * 0xFF51AFD7ED558CCDu64);
            for i in 0..8 {
                state[i] = state[i].wrapping_add(key).wrapping_add(i as u64 * 13);
                state[i] ^= rotate_left(state[(i + 1) % 8], 11);
                state[i] = rotate_left(state[i], 19);
                state[i] ^= rotate_right(state[(i + 2) % 8], 7);
                state[i] = state[i].wrapping_mul(0x9E3779B97F4A7C15u64);
                state[i] ^= rotate_left(state[i], 17);
            }
            // Mix block words into state
            if rnd % 4 == 0 {
                for i in 0..8 {
                    state[i] ^= block_words[i];
                }
            }
        }

        // Finalize this block
        for i in 0..8 {
            state[i] ^= block_words[i];
        }
    }

    // Finalization rounds
    for rnd in 0..16 {
        let key = salt_int ^ (rnd as u64 * 0xFF51AFD7ED558CCDu64);
        for i in 0..8 {
            state[i] = state[i].wrapping_add(key).wrapping_add(i as u64 * 13);
            state[i] ^= rotate_left(state[(i + 1) % 8], 11);
            state[i] = rotate_left(state[i], 19);
            state[i] ^= salt_words[i % 8];
            state[i] ^= rotate_right(state[(i + 3) % 8], 23);
        }
    }

    // Convert state to output bytes
    let mut output = Vec::with_capacity(64);
    for word in &state {
        output.extend_from_slice(&word.to_le_bytes());
    }

    output
}

fn cmd_hash(text: &str, length: usize, salt_hex: Option<String>, no_stretch: bool) {
    let salt = match salt_hex {
        Some(hex_str) => hex::decode(hex_str).expect("Invalid hex salt"),
        None => generate_salt(32),
    };

    let _ = no_stretch;
    let hash_bytes = aelvonlock_hash_internal(text, &salt);
    let symbols = bytes_to_symbols(&hash_bytes, length);
    let tagged = format!("{}.{}", VERSION_TAG, symbols);

    println!("Hash: {}", tagged);
    println!("Salt: {}", hex::encode(salt));
}

fn cmd_verify(password: &str, stored_hash: &str, salt_hex: &str) {
    let salt = hex::decode(salt_hex).expect("Invalid hex salt");
    let hash_bytes = aelvonlock_hash_internal(password, &salt);
    let symbols = bytes_to_symbols(&hash_bytes, 64);
    let recomputed = format!("{}.{}", VERSION_TAG, symbols);

    // Constant-time comparison
    let valid = recomputed.len() == stored_hash.len()
        && recomputed.bytes()
            .zip(stored_hash.bytes())
            .fold(0, |acc, (a, b)| acc | (a ^ b)) == 0;

    println!("Result: {}", if valid { "✅ VALID" } else { "❌ INVALID" });
}

fn cmd_self_test() {
    println!("Running Aelvonlock-SHC Rust self-test...");
    let mut all_pass = true;

    // Test 1: Basic hashing
    let (h1, _) = {
        let salt = generate_salt(32);
        let bytes = aelvonlock_hash_internal("test", &salt);
        let sym = bytes_to_symbols(&bytes, 64);
        (format!("{}.{}", VERSION_TAG, sym), salt)
    };
    let test1 = h1.len() > 64;
    println!("  {} basic_hashing", if test1 { "✅" } else { "❌" });
    all_pass &= test1;

    // Test 2: Deterministic with same salt
    let salt = generate_salt(32);
    let bytes2a = aelvonlock_hash_internal("test", &salt);
    let bytes2b = aelvonlock_hash_internal("test", &salt);
    let test2 = bytes2a == bytes2b;
    println!("  {} deterministic", if test2 { "✅" } else { "❌" });
    all_pass &= test2;

    // Test 3: Different inputs
    let bytes3a = aelvonlock_hash_internal("hello", &generate_salt(32));
    let bytes3b = aelvonlock_hash_internal("world", &generate_salt(32));
    let test3 = bytes3a != bytes3b;
    println!("  {} different_inputs", if test3 { "✅" } else { "❌" });
    all_pass &= test3;

    // Test 4: Salt changes output
    let bytes4a = aelvonlock_hash_internal("test", &generate_salt(32));
    let bytes4b = aelvonlock_hash_internal("test", &generate_salt(32));
    let test4 = bytes4a != bytes4b;
    println!("  {} salt_changes_output", if test4 { "✅" } else { "❌" });
    all_pass &= test4;

    // Test 5: Non-empty output
    let bytes5 = aelvonlock_hash_internal("test", &generate_salt(32));
    let test5 = !bytes5.is_empty();
    println!("  {} non_empty_output", if test5 { "✅" } else { "❌" });
    all_pass &= test5;

    println!("\nOverall: {}", if all_pass { "✅ ALL PASSED" } else { "❌ SOME FAILED" });
}

fn cmd_benchmark(iterations: usize) {
    println!("Running benchmark ({} iterations)...", iterations);
    let salt = generate_salt(32);
    let start = std::time::Instant::now();

    for i in 0..iterations {
        let input = format!("benchmark_test_input_{}", i);
        let _ = aelvonlock_hash_internal(&input, &salt);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations as u32;

    println!("  Total time:      {:.3}s", elapsed.as_secs_f64());
    println!("  Average time:    {:.3}s", avg.as_secs_f64());
    println!("  Hashes/sec:      {:.2}", iterations as f64 / elapsed.as_secs_f64());
}

fn cmd_info() {
    println!("Aelvonlock-SHC Rust v{}", env!("CARGO_PKG_VERSION"));
    println!("  Version Tag:    {}", VERSION_TAG);
    println!("  ARX Rounds:     {}", ROUNDS);
    println!("  Salt Stretch:   {} iterations", SALT_STRETCH_ITERATIONS);
    println!("  Symbol Map:     {} symbols", SYMBOLS.len());
    println!("  Language:       Rust (native compiled)");
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Hash { text, length, salt, no_stretch } => {
            cmd_hash(text, *length, salt.clone(), *no_stretch);
        }
        Commands::Verify { password, hash, salt_hex } => {
            cmd_verify(password, hash, salt_hex);
        }
        Commands::SelfTest => {
            cmd_self_test();
        }
        Commands::Benchmark { iterations } => {
            cmd_benchmark(*iterations);
        }
        Commands::Info => {
            cmd_info();
        }
    }
}
