#!/usr/bin/env python3
"""
AES Parallel Block Cipher: CPU vs GPU Implementation
Demonstrates AES-128 encryption/decryption with performance comparison
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple

# Try to import CuPy for GPU operations
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU is available!")
    print(f"CuPy version: {cp.__version__}")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU not available. Running CPU-only mode.")

# AES S-Box
SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

# AES Inverse S-Box
INV_SBOX = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
], dtype=np.uint8)

# Round constant for key expansion
RCON = np.array([
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
    0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39
], dtype=np.uint8)


def gmul(a, b):
    """Galois Field multiplication"""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set:
            a ^= 0x1b
        b >>= 1
    return p % 256


class AES_CPU:
    """CPU implementation of AES-128 encryption and decryption"""
    
    def __init__(self, key: np.ndarray):
        """Initialize with a 16-byte key"""
        if len(key) != 16:
            raise ValueError("Key must be 16 bytes for AES-128")
        self.key = key.astype(np.uint8)
        self.round_keys = self._key_expansion()
    
    def _key_expansion(self) -> np.ndarray:
        """Expand the key for AES rounds"""
        round_keys = np.zeros((11, 4, 4), dtype=np.uint8)
        round_keys[0] = self.key.reshape(4, 4)
        
        for i in range(1, 11):
            # Get previous round key
            prev = round_keys[i-1].copy()
            
            # Rotate and substitute last column
            temp = np.roll(prev[:, 3], -1)
            temp = SBOX[temp]
            
            # XOR with first column and round constant
            round_keys[i, :, 0] = prev[:, 0] ^ temp ^ np.array([RCON[i], 0, 0, 0], dtype=np.uint8)
            
            # Generate remaining columns
            for j in range(1, 4):
                round_keys[i, :, j] = round_keys[i, :, j-1] ^ prev[:, j]
        
        return round_keys
    
    def _sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """Apply S-Box substitution"""
        return SBOX[state]
    
    def _inv_sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """Apply inverse S-Box substitution"""
        return INV_SBOX[state]
    
    def _shift_rows(self, state: np.ndarray) -> np.ndarray:
        """Shift rows transformation"""
        result = state.copy()
        result[1] = np.roll(state[1], -1)
        result[2] = np.roll(state[2], -2)
        result[3] = np.roll(state[3], -3)
        return result
    
    def _inv_shift_rows(self, state: np.ndarray) -> np.ndarray:
        """Inverse shift rows transformation"""
        result = state.copy()
        result[1] = np.roll(state[1], 1)
        result[2] = np.roll(state[2], 2)
        result[3] = np.roll(state[3], 3)
        return result
    
    def _mix_columns(self, state: np.ndarray) -> np.ndarray:
        """Mix columns transformation"""
        result = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            s0, s1, s2, s3 = state[:, i]
            result[0, i] = gmul(2, s0) ^ gmul(3, s1) ^ s2 ^ s3
            result[1, i] = s0 ^ gmul(2, s1) ^ gmul(3, s2) ^ s3
            result[2, i] = s0 ^ s1 ^ gmul(2, s2) ^ gmul(3, s3)
            result[3, i] = gmul(3, s0) ^ s1 ^ s2 ^ gmul(2, s3)
        return result
    
    def _inv_mix_columns(self, state: np.ndarray) -> np.ndarray:
        """Inverse mix columns transformation"""
        result = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4):
            s0, s1, s2, s3 = state[:, i]
            result[0, i] = gmul(14, s0) ^ gmul(11, s1) ^ gmul(13, s2) ^ gmul(9, s3)
            result[1, i] = gmul(9, s0) ^ gmul(14, s1) ^ gmul(11, s2) ^ gmul(13, s3)
            result[2, i] = gmul(13, s0) ^ gmul(9, s1) ^ gmul(14, s2) ^ gmul(11, s3)
            result[3, i] = gmul(11, s0) ^ gmul(13, s1) ^ gmul(9, s2) ^ gmul(14, s3)
        return result
    
    def _add_round_key(self, state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
        """XOR state with round key"""
        return state ^ round_key
    
    def encrypt_block(self, block: np.ndarray) -> np.ndarray:
        """Encrypt a single 16-byte block"""
        state = block.reshape(4, 4).copy()
        
        # Initial round
        state = self._add_round_key(state, self.round_keys[0])
        
        # Main rounds
        for i in range(1, 10):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, self.round_keys[i])
        
        # Final round (no mix columns)
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, self.round_keys[10])
        
        return state.flatten()
    
    def decrypt_block(self, block: np.ndarray) -> np.ndarray:
        """Decrypt a single 16-byte block"""
        state = block.reshape(4, 4).copy()
        
        # Initial round
        state = self._add_round_key(state, self.round_keys[10])
        
        # Main rounds
        for i in range(9, 0, -1):
            state = self._inv_shift_rows(state)
            state = self._inv_sub_bytes(state)
            state = self._add_round_key(state, self.round_keys[i])
            state = self._inv_mix_columns(state)
        
        # Final round (no inv mix columns)
        state = self._inv_shift_rows(state)
        state = self._inv_sub_bytes(state)
        state = self._add_round_key(state, self.round_keys[0])
        
        return state.flatten()
    
    def encrypt(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Encrypt data (multiple blocks)"""
        # Pad data to multiple of 16 bytes
        if len(data) % 16 != 0:
            pad_len = 16 - (len(data) % 16)
            data = np.concatenate([data, np.zeros(pad_len, dtype=np.uint8)])
        
        n_blocks = len(data) // 16
        result = np.zeros_like(data)
        
        start_time = time.time()
        for i in range(n_blocks):
            block = data[i*16:(i+1)*16]
            result[i*16:(i+1)*16] = self.encrypt_block(block)
        end_time = time.time()
        
        return result, end_time - start_time
    
    def decrypt(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Decrypt data (multiple blocks)"""
        n_blocks = len(data) // 16
        result = np.zeros_like(data)
        
        start_time = time.time()
        for i in range(n_blocks):
            block = data[i*16:(i+1)*16]
            result[i*16:(i+1)*16] = self.decrypt_block(block)
        end_time = time.time()
        
        return result, end_time - start_time


if GPU_AVAILABLE:
    class AES_GPU:
        """GPU implementation of AES-128 using CuPy with vectorized operations"""
        
        def __init__(self, key: np.ndarray):
            """Initialize with a 16-byte key"""
            # Store original key
            self.key = key.astype(np.uint8)
            
            # Use CPU implementation for key expansion
            cpu_aes = AES_CPU(key)
            self.round_keys = cpu_aes.round_keys
            
            # Copy lookup tables to GPU
            self.d_sbox = cp.array(SBOX)
            self.d_inv_sbox = cp.array(INV_SBOX)
            
            # Pre-allocate all round keys on GPU
            self.d_round_keys = [cp.array(self.round_keys[i]) for i in range(11)]
        
        def encrypt(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
            """Encrypt data using GPU vectorized operations"""
            # Pad data
            if len(data) % 16 != 0:
                pad_len = 16 - (len(data) % 16)
                data = np.concatenate([data, np.zeros(pad_len, dtype=np.uint8)])
            
            n_blocks = len(data) // 16
            
            # Reshape data into blocks and copy to GPU
            blocks = data.reshape(n_blocks, 16)
            d_blocks = cp.array(blocks)
            
            # Start timing
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            
            # Reshape to (n_blocks, 4, 4) for processing
            d_state = d_blocks.reshape(n_blocks, 4, 4)
            
            # Initial AddRoundKey
            d_state = d_state ^ self.d_round_keys[0][None, :, :]
            
            # Main rounds - using CPU operations since GPU doesn't have proper GF multiplication
            # Note: For production, implement proper GF multiplication tables on GPU
            for round_num in range(1, 10):
                # SubBytes
                d_state = self.d_sbox[d_state]
                
                # ShiftRows (in-place using temporary)
                temp = d_state.copy()
                d_state[:, 1, :] = cp.roll(temp[:, 1, :], -1, axis=1)
                d_state[:, 2, :] = cp.roll(temp[:, 2, :], -2, axis=1)
                d_state[:, 3, :] = cp.roll(temp[:, 3, :], -3, axis=1)
                
                # MixColumns - Use CPU for correct GF multiplication
                # Convert to CPU, apply mix columns, convert back
                state_cpu = cp.asnumpy(d_state)
                for b in range(n_blocks):
                    state_cpu[b] = self._mix_columns_cpu(state_cpu[b])
                d_state = cp.array(state_cpu)
                
                # AddRoundKey
                d_state = d_state ^ self.d_round_keys[round_num][None, :, :]
            
            # Final round (no MixColumns)
            d_state = self.d_sbox[d_state]
            
            # ShiftRows (in-place using temporary)
            temp = d_state.copy()
            d_state[:, 1, :] = cp.roll(temp[:, 1, :], -1, axis=1)
            d_state[:, 2, :] = cp.roll(temp[:, 2, :], -2, axis=1)
            d_state[:, 3, :] = cp.roll(temp[:, 3, :], -3, axis=1)
            
            # AddRoundKey
            d_state = d_state ^ self.d_round_keys[10][None, :, :]
            
            # End timing
            end.record()
            end.synchronize()
            gpu_time = cp.cuda.get_elapsed_time(start, end) / 1000.0  # Convert to seconds
            
            # Copy result back and reshape
            result = cp.asnumpy(d_state.reshape(n_blocks * 16))
            
            return result, gpu_time
        
        def _mix_columns_cpu(self, state: np.ndarray) -> np.ndarray:
            """Mix columns helper using CPU operations"""
            result = np.zeros((4, 4), dtype=np.uint8)
            for i in range(4):
                s0, s1, s2, s3 = state[:, i]
                result[0, i] = gmul(2, s0) ^ gmul(3, s1) ^ s2 ^ s3
                result[1, i] = s0 ^ gmul(2, s1) ^ gmul(3, s2) ^ s3
                result[2, i] = s0 ^ s1 ^ gmul(2, s2) ^ gmul(3, s3)
                result[3, i] = gmul(3, s0) ^ s1 ^ s2 ^ gmul(2, s3)
            return result
        
        def decrypt(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
            """Decrypt data using CPU (proper key usage)"""
            # Use the original key for decryption
            cpu_aes = AES_CPU(self.key)
            return cpu_aes.decrypt(data)


def benchmark_aes(data_sizes):
    """Benchmark AES encryption for different data sizes"""
    # Generate a random 16-byte key
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    
    # Initialize CPU and GPU implementations
    cpu_aes = AES_CPU(key)
    if GPU_AVAILABLE:
        gpu_aes = AES_GPU(key)
    
    results = {
        'sizes': [],
        'cpu_times': [],
        'gpu_times': [],
        'speedups': []
    }
    
    for size in data_sizes:
        print(f"\nTesting with {size} bytes...")
        
        # Generate random data
        data = np.random.randint(0, 256, size, dtype=np.uint8)
        
        # CPU encryption
        encrypted_cpu, cpu_time = cpu_aes.encrypt(data.copy())
        print(f"  CPU time: {cpu_time:.6f} seconds")
        
        # GPU encryption (if available)
        if GPU_AVAILABLE:
            encrypted_gpu, gpu_time = gpu_aes.encrypt(data.copy())
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  GPU time: {gpu_time:.6f} seconds")
            print(f"  Speedup: {speedup:.2f}x")
            
            results['gpu_times'].append(gpu_time)
            results['speedups'].append(speedup)
        else:
            results['gpu_times'].append(0)
            results['speedups'].append(0)
        
        results['sizes'].append(size)
        results['cpu_times'].append(cpu_time)
    
    return results


def test_correctness():
    """Test that encryption and decryption work correctly"""
    print("Testing AES Correctness...")
    print("="*50)
    
    # Test data
    key = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8)
    
    test_message = b"Hello, World! This is a test message for AES encryption."
    plaintext = np.frombuffer(test_message, dtype=np.uint8)
    
    # CPU test
    print("\nCPU Implementation:")
    cpu_aes = AES_CPU(key)
    encrypted, enc_time = cpu_aes.encrypt(plaintext.copy())
    decrypted, dec_time = cpu_aes.decrypt(encrypted)
    
    # Remove padding
    decrypted = decrypted[:len(plaintext)]
    
    print(f"Original:  {plaintext[:20]}...")
    print(f"Encrypted: {encrypted[:20]}...")
    print(f"Decrypted: {decrypted[:20]}...")
    print(f"Match: {np.array_equal(plaintext, decrypted)}")
    print(f"Encryption time: {enc_time:.6f}s")
    print(f"Decryption time: {dec_time:.6f}s")
    
    if GPU_AVAILABLE:
        print("\nGPU Implementation:")
        gpu_aes = AES_GPU(key)
        encrypted_gpu, enc_time_gpu = gpu_aes.encrypt(plaintext.copy())
        print(f"GPU Encrypted: {encrypted_gpu[:20]}...")
        print(f"GPU Encryption time: {enc_time_gpu:.6f}s")
        if enc_time_gpu > 0:
            print(f"GPU speedup: {enc_time/enc_time_gpu:.2f}x")


def visualize_results(results):
    """Create visualizations of benchmark results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Execution Time Comparison
    ax1 = axes[0, 0]
    ax1.plot(results['sizes'], results['cpu_times'], 'o-', label='CPU', linewidth=2)
    if GPU_AVAILABLE and any(results['gpu_times']):
        ax1.plot(results['sizes'], results['gpu_times'], 's-', label='GPU', linewidth=2)
    ax1.set_xlabel('Data Size (bytes)', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('AES Encryption: CPU vs GPU Time', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    ax2 = axes[0, 1]
    if GPU_AVAILABLE and any(results['speedups']):
        ax2.plot(results['sizes'], results['speedups'], 'o-', color='green', linewidth=2)
        ax2.axhline(y=1, color='r', linestyle='--', label='No speedup')
    ax2.set_xlabel('Data Size (bytes)', fontsize=12)
    ax2.set_ylabel('Speedup (CPU/GPU)', fontsize=12)
    ax2.set_title('GPU Speedup over CPU', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Throughput
    ax3 = axes[1, 0]
    cpu_throughput = [size/time/1e6 for size, time in zip(results['sizes'], results['cpu_times'])]
    ax3.plot(results['sizes'], cpu_throughput, 'o-', label='CPU', linewidth=2)
    if GPU_AVAILABLE and any(results['gpu_times']):
        gpu_throughput = [size/time/1e6 if time > 0 else 0 for size, time in zip(results['sizes'], results['gpu_times'])]
        ax3.plot(results['sizes'], gpu_throughput, 's-', label='GPU', linewidth=2)
    ax3.set_xlabel('Data Size (bytes)', fontsize=12)
    ax3.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax3.set_title('AES Encryption Throughput', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    table_data = []
    for i, size in enumerate(results['sizes']):
        row = [
            f"{size:.0e}",
            f"{results['cpu_times'][i]:.4f}s",
        ]
        if GPU_AVAILABLE:
            row.extend([
                f"{results['gpu_times'][i]:.4f}s",
                f"{results['speedups'][i]:.2f}x"
            ])
        table_data.append(row)
    
    headers = ['Size', 'CPU Time']
    if GPU_AVAILABLE:
        headers.extend(['GPU Time', 'Speedup'])
    
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('aes_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'aes_performance_comparison.png'")
    plt.show()


if __name__ == "__main__":
    # Test correctness first
    test_correctness()
    
    print("\n" + "="*50)
    print("Starting AES Performance Benchmark...")
    print("="*50)
    
    # Test with data sizes from 10^1 to 10^6 bytes (limited for practical testing)
    data_sizes = [10**i for i in range(1, 7)]  # 10, 100, 1K, 10K, 100K, 1M bytes
    
    benchmark_results = benchmark_aes(data_sizes)
    
    # Visualize results
    visualize_results(benchmark_results)
    
    print("\n" + "="*50)
    print("Benchmark complete!")
    print("="*50)
