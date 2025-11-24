#!/usr/bin/env python3
"""
Quick Start Example for AES CPU/GPU Implementation

This script demonstrates how to use the AES encryption implementation.
"""

import numpy as np
from aes_benchmark import AES_CPU, GPU_AVAILABLE

if GPU_AVAILABLE:
    from aes_benchmark import AES_GPU

def example_basic_usage():
    """Basic encryption and decryption example"""
    print("="*60)
    print("Example 1: Basic Encryption and Decryption")
    print("="*60)
    
    # Create a 16-byte key (AES-128)
    key = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8)
    
    # Original message
    message = b"Hello, World! This is a secret message."
    plaintext = np.frombuffer(message, dtype=np.uint8)
    
    # Initialize AES cipher
    aes = AES_CPU(key)
    
    # Encrypt
    print(f"\nOriginal message: {message.decode('utf-8')}")
    encrypted, enc_time = aes.encrypt(plaintext.copy())
    print(f"Encrypted (hex): {encrypted[:32].tobytes().hex()}...")
    print(f"Encryption time: {enc_time:.6f} seconds")
    
    # Decrypt
    decrypted, dec_time = aes.decrypt(encrypted)
    decrypted = decrypted[:len(plaintext)]  # Remove padding
    decrypted_message = decrypted.tobytes().decode('utf-8')
    print(f"Decrypted message: {decrypted_message}")
    print(f"Decryption time: {dec_time:.6f} seconds")
    print(f"Match: {message.decode('utf-8') == decrypted_message}")


def example_random_key():
    """Example with randomly generated key"""
    print("\n" + "="*60)
    print("Example 2: Using Random Key")
    print("="*60)
    
    # Generate random 16-byte key
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    print(f"Random key (hex): {key.tobytes().hex()}")
    
    # Message to encrypt
    message = b"Secret data with random key!"
    plaintext = np.frombuffer(message, dtype=np.uint8)
    
    # Encrypt and decrypt
    aes = AES_CPU(key)
    encrypted, _ = aes.encrypt(plaintext.copy())
    decrypted, _ = aes.decrypt(encrypted)
    decrypted = decrypted[:len(plaintext)]
    
    print(f"Original: {message.decode('utf-8')}")
    print(f"Decrypted: {decrypted.tobytes().decode('utf-8')}")
    print(f"Success: {np.array_equal(plaintext, decrypted)}")


def example_large_data():
    """Example with larger data"""
    print("\n" + "="*60)
    print("Example 3: Encrypting Large Data")
    print("="*60)
    
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    
    # Generate 100KB of random data
    data_size = 100 * 1024  # 100 KB
    data = np.random.randint(0, 256, data_size, dtype=np.uint8)
    print(f"Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
    
    # CPU encryption
    aes_cpu = AES_CPU(key)
    encrypted_cpu, cpu_time = aes_cpu.encrypt(data.copy())
    print(f"CPU encryption time: {cpu_time:.4f} seconds")
    print(f"CPU throughput: {data_size/cpu_time/1024/1024:.2f} MB/s")
    
    # GPU encryption (if available)
    if GPU_AVAILABLE:
        aes_gpu = AES_GPU(key)
        encrypted_gpu, gpu_time = aes_gpu.encrypt(data.copy())
        print(f"GPU encryption time: {gpu_time:.4f} seconds")
        print(f"GPU throughput: {data_size/gpu_time/1024/1024:.2f} MB/s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU not available for comparison")
    
    # Verify decryption
    decrypted, _ = aes_cpu.decrypt(encrypted_cpu)
    decrypted = decrypted[:data_size]
    print(f"Decryption successful: {np.array_equal(data, decrypted)}")


def example_file_like_encryption():
    """Example simulating file encryption"""
    print("\n" + "="*60)
    print("Example 4: File-like Encryption (Chunks)")
    print("="*60)
    
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    aes = AES_CPU(key)
    
    # Simulate file content
    file_content = b"This is line 1 of the file.\n"
    file_content += b"This is line 2 of the file.\n"
    file_content += b"This is line 3 of the file.\n"
    file_content += b"End of file."
    
    plaintext = np.frombuffer(file_content, dtype=np.uint8)
    
    print(f"Original content ({len(file_content)} bytes):")
    print(file_content.decode('utf-8'))
    
    # Encrypt
    encrypted, enc_time = aes.encrypt(plaintext.copy())
    print(f"\nEncrypted to {len(encrypted)} bytes in {enc_time:.6f}s")
    
    # Decrypt
    decrypted, dec_time = aes.decrypt(encrypted)
    decrypted = decrypted[:len(plaintext)]
    
    print(f"\nDecrypted content:")
    print(decrypted.tobytes().decode('utf-8'))


def example_key_from_password():
    """Example deriving key from password (simplified)"""
    print("\n" + "="*60)
    print("Example 5: Key from Password (Simplified)")
    print("="*60)
    
    # Note: This is simplified. Use proper KDF (PBKDF2, Argon2) in production
    password = "my_secure_password_2024"
    
    # Simple hash to create 16-byte key (NOT SECURE for production)
    import hashlib
    key_hash = hashlib.md5(password.encode()).digest()
    key = np.frombuffer(key_hash, dtype=np.uint8)
    
    print(f"Password: {password}")
    print(f"Derived key (hex): {key.tobytes().hex()}")
    print("⚠️  Warning: Use proper KDF (PBKDF2, Argon2) in production!")
    
    # Use the key
    message = b"Message encrypted with password-derived key"
    plaintext = np.frombuffer(message, dtype=np.uint8)
    
    aes = AES_CPU(key)
    encrypted, _ = aes.encrypt(plaintext.copy())
    decrypted, _ = aes.decrypt(encrypted)
    decrypted = decrypted[:len(plaintext)]
    
    print(f"\nOriginal: {message.decode('utf-8')}")
    print(f"Decrypted: {decrypted.tobytes().decode('utf-8')}")
    print(f"Success: {np.array_equal(plaintext, decrypted)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AES Encryption Examples")
    print("="*60)
    
    # Run all examples
    example_basic_usage()
    example_random_key()
    example_large_data()
    example_file_like_encryption()
    example_key_from_password()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nFor full benchmarking, run: python aes_benchmark.py")
    print("For interactive use, see: AES_CPU_GPU_Comparison.ipynb")
