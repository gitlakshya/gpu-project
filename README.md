# AES Parallel Block Cipher: CPU vs GPU Implementation

This project implements AES-128 encryption and decryption on both CPU and GPU, demonstrating the speedup achieved through GPU parallelization using CuPy.

## Overview

This implementation showcases:
- **CPU Implementation**: Pure Python/NumPy implementation of AES-128
- **GPU Implementation**: CuPy-based GPU-accelerated version
- **Performance Benchmarking**: Comparison across data sizes from 10¹ to 10⁶ bytes
- **Visualization**: Graphs showing speedup and throughput improvements

## Files

- `AES_CPU_GPU_Comparison.ipynb` - Jupyter notebook with complete implementation and benchmarks (Google Colab compatible)
- `aes_benchmark.py` - Python script version for command-line execution
- `README.md` - This file

## Requirements

### For CPU-only execution:
```bash
pip install numpy matplotlib
```

### For GPU execution (Google Colab or CUDA-enabled environment):
```bash
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
pip install numpy matplotlib
```

## Usage

### Using Jupyter Notebook (Recommended for Google Colab)

1. Open `AES_CPU_GPU_Comparison.ipynb` in Google Colab or Jupyter
2. If using Google Colab, enable GPU runtime:
   - Runtime → Change runtime type → Hardware accelerator → GPU
3. Run all cells sequentially

### Using Python Script

```bash
python aes_benchmark.py
```

This will:
1. Test correctness of encryption/decryption
2. Benchmark performance across multiple data sizes
3. Generate visualization graphs
4. Save results to `aes_performance_comparison.png`

## AES-128 Implementation Details

### CPU Implementation
The CPU implementation includes:
- **Key Expansion**: Generates 11 round keys from the original 16-byte key
- **SubBytes**: S-Box substitution
- **ShiftRows**: Row shifting transformation
- **MixColumns**: Column mixing in Galois Field
- **AddRoundKey**: XOR with round key

### GPU Implementation
The GPU implementation uses CuPy for:
- **Vectorized Operations**: Process multiple blocks in parallel
- **GPU Memory Management**: Efficient data transfer between CPU and GPU
- **Parallel Block Processing**: Each block encrypted independently

### Encryption Flow
```
Plaintext → Padding (if needed) → Split into 16-byte blocks → 
For each block:
  Initial AddRoundKey →
  9 Rounds (SubBytes, ShiftRows, MixColumns, AddRoundKey) →
  Final Round (SubBytes, ShiftRows, AddRoundKey) →
Ciphertext
```

## Performance Characteristics

### Expected Speedup
- **Small data (<1KB)**: GPU overhead may make CPU faster
- **Medium data (1KB-100KB)**: GPU starts showing benefits (2-5x speedup)
- **Large data (>100KB)**: Significant GPU speedup (5-50x depending on hardware)

### Benchmark Data Sizes
- 10¹ bytes (10 bytes)
- 10² bytes (100 bytes)
- 10³ bytes (1 KB)
- 10⁴ bytes (10 KB)
- 10⁵ bytes (100 KB)
- 10⁶ bytes (1 MB)

## Example Output

```
Testing AES Correctness...
==================================================

CPU Implementation:
Original:  [ 72 101 108 108 111  44  32  87 111 114 108 100  33  32  84 104 105 115  32 105]...
Match: True
Encryption time: 0.001234s
Decryption time: 0.001189s

GPU Implementation:
GPU Encryption time: 0.000234s
GPU speedup: 5.27x

Starting AES Performance Benchmark...
==================================================

Testing with 10 bytes...
  CPU time: 0.000050 seconds
  GPU time: 0.000012 seconds
  Speedup: 4.17x

Testing with 1000000 bytes...
  CPU time: 5.234567 seconds
  GPU time: 0.123456 seconds
  Speedup: 42.38x
```

## Educational Notes

### Block Cipher Mode
This implementation uses ECB (Electronic Codebook) mode for simplicity. For production use:
- Use CBC, CTR, or GCM modes
- Implement proper IV (Initialization Vector) handling
- Add authentication (HMAC or AEAD)

### Security Considerations
This is an educational implementation. For production:
- Use established libraries (PyCryptodome, cryptography)
- Implement constant-time operations to prevent timing attacks
- Proper key management and storage
- PKCS#7 padding scheme

### Limitations
- Simplified MixColumns operation in GPU version for demonstration
- ECB mode (not recommended for production)
- No authentication/integrity checking
- Not optimized for maximum performance

## Google Colab Setup

To run in Google Colab:

1. Create a new notebook or upload `AES_CPU_GPU_Comparison.ipynb`
2. Enable GPU:
   ```python
   # Check GPU availability
   !nvidia-smi
   ```
3. Install CuPy:
   ```python
   !pip install cupy-cuda11x
   ```
4. Run the benchmark cells

## Visualization Output

The benchmark generates a 4-panel visualization:
1. **Execution Time Comparison**: Log-log plot of CPU vs GPU times
2. **Speedup Chart**: GPU speedup factor across data sizes
3. **Throughput Comparison**: MB/s processed by CPU and GPU
4. **Summary Table**: Detailed performance metrics

## License

This is an educational project for demonstrating GPU acceleration concepts.

## References

- [FIPS 197: Advanced Encryption Standard (AES)](https://csrc.nist.gov/publications/detail/fips/197/final)
- [CuPy Documentation](https://docs.cupy.dev/)
- [NumPy Documentation](https://numpy.org/doc/)

## Author

Created as an assignment for GPU Research coursework to demonstrate:
- AES block cipher implementation
- CPU to GPU parallelization
- Performance analysis and visualization
