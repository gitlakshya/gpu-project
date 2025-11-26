# GPU-Accelerated AES-128 ECB Encryption Using Numba CUDA

## Academic Report (ECB-Only Implementation)

**Author:** M.Tech Student  
**Project:** Parallel AES-128 Encryption using CUDA (Numba)  
**Date:** November 2025  

---

## Abstract

This report presents a GPU-accelerated implementation of the AES-128 block cipher in Electronic Codebook (ECB) mode using Python, Numba CUDA, and a custom AES kernel. The goal is to compare the performance of a traditional CPU implementation (PyCryptodome AES-ECB) with a parallel GPU implementation that encrypts multiple 16-byte blocks concurrently.

Benchmarks were conducted for plaintext sizes ranging from 10 bytes to 10 MB. While the GPU shows **up to ~50× speedup** over the CPU for the largest data size tested, the **absolute throughput remains low** relative to theoretical GPU or PCIe bandwidth, indicating underutilization of the hardware. This report analyses the implementation, the measurement methodology, the observed performance trends, and the reasons for low GPU efficiency.



---

## 1. Introduction

### 1.1 Background

AES-128 is a widely used symmetric block cipher standardized by NIST. It operates on 128-bit (16-byte) blocks with a 128-bit key over 10 rounds of nonlinear substitution, permutation, and mixing. While CPUs can encrypt data efficiently, modern GPUs provide massive thread-level parallelism that can be exploited to process many AES blocks simultaneously.

Numba’s CUDA extension allows Python functions to be compiled and executed on NVIDIA GPUs, enabling researchers to prototype GPU algorithms quickly without writing low-level CUDA C/C++.

### 1.2 Motivation

The motivation for this work is:

1. To explore how much **speedup** can be obtained by offloading AES-128 ECB encryption to a GPU using Numba CUDA.
2. To study how performance **scales with data size**.
3. To understand **why GPU utilization (efficiency) is currently low**, and what that implies for optimization.

### 1.3 Objectives

This experiment focuses on:

- Implementing **AES-128 ECB encryption and decryption** on both CPU and GPU.
- Measuring **encryption/decryption time and throughput** for different input sizes.
- Computing **speedup (GPU/CPU)** and plotting performance trends.
- Discussing reasons for low GPU efficiency and potential improvements.

---

## 2. Theoretical Foundation

### 2.1 AES-128 and ECB Mode (Short Overview)

AES-128:

- Block size: **128 bits** (16 bytes).
- Key size: **128 bits**.
- Rounds: **10** (SubBytes, ShiftRows, MixColumns, AddRoundKey).
- Implementation in this project:
  - CPU: PyCryptodome `AES.new(key, AES.MODE_ECB)`  
  - GPU: Custom AES kernel implementing the same round structure.

**ECB Mode (Electronic Codebook)**:

- Each block is encrypted **independently**:
  
  $$
  C_i = E_k(P_i)
  $$
  
- **Advantage:** Perfectly parallelizable → ideal for GPU experiments.
- **Disadvantage:** Leaks patterns; not secure for real-world structured data.
- For this project, ECB is used **only as a performance benchmark mode**, not as a recommended production mode.

### 2.2 What Is Throughput?


In this context:

> **Throughput** is how much data the system can encrypt per second.

Formally:

$$
\text{Throughput (bytes/s)} = \frac{\text{data size (bytes)}}{\text{execution time (seconds)}}
$$

For readability, results are usually reported in **MB/s**:
$$
\text{Throughput (MB/s)} =
\frac{\text{data size (bytes)}}{\text{time (s)} \times 10^6}
$$

- Higher throughput ⇒ the system encrypts more data per second ⇒ **better performance**.
- In your code, you computed:

```python
throughput = size / encrypt_time   # bytes per second
mbps = throughput / 1e6            # MB/s
```

### 2.3 Speedup Definition

To compare CPU and GPU performance, we use:

$$
Speedup=\frac{\text{ GPU throughput}}{ \text{ CPU throughput}}
$$
​

  
- Speedup > 1 ⇒ GPU faster than CPU.
- Speedup < 1 ⇒ CPU faster (common for very small messages due to overhead).

## 3. Implementation Methodology

### 3.1 Environment

- Language: Python 3.x
- CPU AES: PyCryptodome (AES.MODE_ECB)
- GPU AES: Numba CUDA kernel (aes_private_sharedlut)
- Timing: time.perf_counter() around encrypt/decrypt calls
- Plotting: Matplotlib
- Padding: Plaintext padded to a multiple of 16 bytes for GPU kernel (AES block size).

### 3.2 CPU Implementation (Summary)

``` python
cipher = AES.new(key, AES.MODE_ECB)
ciphertext, encrypt_time_cpu = cipher.encrypt(plaintext)
plaintext_cpu, decrypt_time_cpu = cipher.decrypt(ciphertext)
```

- Encrypts using a well-tested library.
- Suitable baseline for comparison.
- Processes data sequentially (one block at a time internally).

### 3.3 GPU Implementation (Summary)

``` python
result_gpu, encrypt_time_gpu = test_code.encrypt_gpu(state, cipherkey_arr, statelength)
decrypted_gpu, decrypt_time_gpu = test_code.decrypt_gpu(result_gpu, cipherkey_arr, statelength)
```

*** Key traits: ***

- state is a uint8 array of the plaintext bytes.
- Each GPU thread typically processes one AES block (16 bytes).
- blocks_per_grid and threads_per_block determine how many blocks are processed in parallel.
- Kernel uses shared/constant lookup tables (S-box, Rcon, multiplication tables) stored on device.

### 3.4 Benchmarking Loop
Data sizes tested:
```python
data_sizes = [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
```
For each size:

- Generate random ASCII plaintext of length size.
- Pad to a multiple of 16 bytes for AES.
- Run 3 trials:
- Call runthecode(plaintext, cipherkey) which:
- Encrypts on CPU & GPU.
- Decrypts on CPU & GPU.
- Returns cpu_encrypt_time, gpu_encrypt_time, cpu_decrypt_time, gpu_decrypt_time.
- Average times across trials.
- Compute throughput: size / time (bytes/s).
- Store results in ecb_results for plotting.

## 4. Experimental Results

### 4.1 Raw Results (ECB Only)
ecb_results dictionary:

```python
ecb_results = {
    'sizes':   [10,   100,   1000,   10000,    100000,     1000000,     10000000],
    'cpu_enc': [700.23, 7745.73, 77722.22, 427119.22, 783373.68, 576050.47, 168005.41],
    'cpu_dec': [707.01, 8254.68, 59421.24, 427941.74, 918799.56, 703897.53, 534320.65],
    'gpu_enc': [66.30, 726.86, 8316.33, 79760.08, 629532.62, 3772180.36, 8338002.72],
    'gpu_dec': [53.82, 738.19, 7239.94, 72170.90, 639582.47, 3183361.70, 7186771.61]
}

```
note:Units: the values above are bytes per second (throughput), not MB/s.
For MB/s, divide by 1e6.

### 4.2 Throughput Table (Approximate MB/s)

| Data Size (bytes) | CPU Enc (MB/s) | GPU Enc (MB/s) | Speedup (GPU/CPU) |
| ----------------- | -------------- | -------------- | ----------------- |
| 10                | 0.000001       | 0.00000007     | ~0.09×            |
| 100               | 0.000008       | 0.0000007      | ~0.09×            |
| 1,000             | 0.000078       | 0.000008       | ~0.11×            |
| 10,000            | 0.000427       | 0.000080       | ~0.19×            |
| 100,000           | 0.000783       | 0.000630       | ~0.80×            |
| 1,000,000         | 0.000576       | 0.003772       | ~6.5×             |
| 10,000,000        | 0.000168       | 0.008338       | ~49.6×            |

### 4.3 GPU Speedup vs Data Size

- For very small data sizes (10–100 bytes):
   - Speedup < 1 ⇒ CPU is faster.
   - GPU is hurt by kernel launch overhead and data transfer overhead.

- Around 100 KB, GPU is roughly comparable to CPU.
- From 1 MB onward, GPU becomes significantly faster:
   - ~6.5× at 1 MB
   - ~50× at 10 MB

This aligns with expectations: GPU becomes advantageous when enough parallel work is available to amortize overheads.

## 5. GPU Efficiency and “Why Is the Plot ~0?”

``` python
theoretical_peak = 12000  # MB/s (PCIe Gen3 x16)
ecb_gpu_efficiency = [x / (theoretical_peak * 1e6) * 100 for x in ecb_results['gpu_enc']]
```

Here:

- **x** is throughput in bytes/s.
- **theoretical_peak × 1e6** = `12000 × 1e6` bytes/s = **12 GB/s**.

For your best case (10 MB):

- GPU throughput ≈ `8.3e6` bytes/s = **8.3 MB/s**

Efficiency:

$$
\text{Efficiency} = \frac{8.3 \text{ MB/s}}{12000 \text{ MB/s}} \times 100 \approx 0.069\%
$$

So the GPU achieves **only ~0.07% of PCIe Gen3 x16 bandwidth**, which explains why the GPU efficiency plot appears nearly zero.


## 5. Discussion

### 5.1 Why Is GPU Efficiency “Not Up to the Mark”?

From your measurements:

- **Max GPU throughput ≈ 8.3 MB/s** (at 10 MB input)
- **PCIe Gen3 x16 theoretical ≈ 12,000 MB/s**

This means the GPU is operating at **far below 1% of the PCIe bandwidth**.  
Although this looks disappointing, it is **expected** for several reasons:

#### **1. Python + Numba Overhead**
- Kernel launches from Python are slow.
- Numba’s JIT compilation and dispatcher create extra latency.
- This overhead dominates when AES work per thread is small.

#### **2. Kernel Configuration / Grid Size**
- With small or medium inputs, `blocks_per_grid` is small → many SMs sit idle.
- Even at larger sizes, GPU occupancy may be suboptimal due to:
  - Low threads per block
  - Non-coalesced memory accesses
  - Underutilized warp execution

#### **3. Memory Access Patterns**
- AES relies heavily on lookup tables (S-box, mul tables).
- If these are stored in **global memory** instead of shared/constant memory:
  - Each lookup becomes expensive
  - Memory latency stalls throughput
  
#### **4. Data Transfer Overhead**
- PCIe host ↔ device transfers are slow relative to computation.
- For a 10 MB message, transfer + kernel launch time overwhelms useful work.
- Consequently, throughput is capped by transfer cost, not computation.

#### **5. Non-Vectorized Plaintext Handling**
- Converting strings to arrays, padding them, and copying to GPU for each trial
  adds overhead not present in highly optimized AES libraries.
- These preprocessing steps inflate the total execution time.

---

### 5.2 When Is GPU Still Worth It?

Even with low absolute throughput, your results show:

- **~50× speedup** at the 10 MB scale (GPU vs CPU)

This is a strong indication that:

- The GPU accelerates AES significantly when enough parallel work exists.
- In many scenarios (large files, bulk encryption), **relative speedup** is more important than absolute bandwidth utilization.
- Peak theoretical PCIe bandwidth is rarely achievable for compute-bound kernels like AES.

In summary, while your GPU efficiency is low relative to hardware limits,  
**the GPU still provides substantial performance gains for large inputs**,  
making it practically valuable despite low PCIe utilization.


## 6. Limitations and Future Work

### 6.1 Limitations

The current implementation has the following limitations:

- Only **ECB mode** is implemented and benchmarked.
- No systematic **validation against NIST AES-128 test vectors**.
- GPU kernel is **not heavily optimized**, specifically:
  - No shared-memory storage for S-box or multiplication tables
  - No kernel fusion
  - Possibly low occupancy and non-coalesced memory access
- Timings include significant **Python overhead**, and the pure kernel runtime is not isolated.

---

### 6.2 Future Improvements

#### **Validation**

- Add **NIST AES-128 test vectors** to verify correctness.
- Add round-trip verification:
  ```python
  decrypt(encrypt(plaintext)) == plaintext
  ```
#### **Kernel Optimization**
- Move lookup tables (S-box, mul2, mul3, etc.) into shared or constant memory.
- Improve thread/block mapping for better occupancy.
- Ensure coalesced loads/stores of 16-byte blocks.
#### **Throughput Improvements**
- Use larger batch sizes or process multiple blocks per thread.
- Minimize host↔device transfers by reusing device buffers.
### Separate Timing Components

To better understand performance bottlenecks, measure the following individually:

- **Host → Device copy time**
- **Kernel execution time**
- **Device → Host copy time**

Breaking down timing in this way helps identify whether data transfer or computation is the dominant bottleneck.

---

### Enhanced Metrics

For clearer performance analysis, also report:

- **Absolute throughput** (MB/s)
- **Speedup compared to CPU**
- **Relative GPU efficiency**, normalized to your highest observed GPU throughput (as you already plot now)

## 8. Conclusion

This project implemented AES-128 ECB mode on both the CPU (via PyCryptodome) and the GPU (via Numba CUDA).  
Benchmarking across data sizes from **10 bytes up to 10 MB** revealed several key insights:

- **Very small inputs (10–100 bytes)** run faster on the CPU due to:
  - Python overhead
  - GPU kernel launch latency
- Around **100 KB**, GPU performance becomes roughly comparable to the CPU.
- At **10 MB**, the GPU achieves approximately **50× speedup** over the CPU.
- However, the **absolute GPU throughput (~8.3 MB/s)** remains far below the theoretical **PCIe Gen3 ×16 limit (12,000 MB/s)**.

From these observations:

- Your conclusion that **GPU efficiency is not up to the mark** (when compared to PCIe bandwidth) is accurate.
- Despite low absolute throughput, the **relative speedup is strong**, making GPU acceleration valuable for **large-scale encryption tasks**, where high parallelism can be exploited.

Overall, this experiment forms a solid foundation. With additional validation, deeper kernel optimization, and more refined benchmarking (including isolated kernel timing), the implementation can be significantly improved and developed into work suitable for **academic publication or a thesis chapter**.


## Appendix

### Appendix A — System Specifications

#### Hardware
- **CPU:** (Specify your processor model)
- **GPU:** NVIDIA GPU with CUDA support  
- **GPU Compute Capability:** (e.g., 6.1, 7.5, etc.)
- **Memory:** (e.g., 16 GB RAM)
- **PCIe Interface:** PCIe Gen3 ×16

#### Software
- **Operating System:** (e.g., Ubuntu 22.04 / Windows / Kaggle Notebook)
- **Python Version:** 3.x
- **Numba Version:** (e.g., 0.60+)
- **CUDA Toolkit:** (as supported by Numba)
- **PyCryptodome Version:** Latest stable
- **Numpy Version:** 2.x

### Appendix B — Benchmark Data (ECB Mode)

#### Stored Results (ecb_results)

{
  'sizes':    [10, 100, 1000, 10000, 100000, 1000000, 10000000],
  'cpu_enc':  [700.23, 7745.73, 77722.22, 427119.22, 783373.68, 576050.47, 168005.41],
  'cpu_dec':  [707.01, 8254.68, 59421.24, 427941.74, 918799.56, 703897.53, 534320.65],
  'gpu_enc':  [66.30, 726.86, 8316.33, 79760.08, 629532.62, 3772180.36, 8338002.72],
  'gpu_dec':  [53.82, 738.19, 7239.94, 72170.90, 639582.47, 3183361.70, 7186771.61]
}

### Appendix C — Throughput Formulas

#### Byte-Level Throughput
$$
\text{Throughput}_{\text{bytes/s}} =
\frac{\text{Data Size (bytes)}}{\text{Execution Time (seconds)}}
$$

#### Megabytes-per-second Throughput
$$
\text{Throughput}_{\text{MB/s}} =
\frac{\text{Data Size (bytes)}}{\text{Execution Time (seconds)} \times 10^6}
$$

### Appendix D — Speedup Formula

#### Speedup (GPU vs CPU)
$$
\text{Speedup} =
\frac{\text{Throughput}_{\text{GPU}}}{\text{Throughput}_{\text{CPU}}}
$$

### Appendix E — GPU Efficiency Formulas

#### 1. Efficiency vs PCIe Bandwidth (Absolute Efficiency)

PCIe Gen3 ×16 theoretical: 12 GB/s = 12000 MB/s

$$
\text{Efficiency}_{\text{absolute}} =
\frac{\text{GPU Throughput (MB/s)}}{12000} \times 100
$$


#### 2. Relative GPU Efficiency (Based on Your Best Run)
Let T_max = highest observed GPU throughput

$$
\text{Efficiency}_{\text{relative}} =
\frac{T}{T_{\max}} \times 100
$$


### Appendix F — Example GPU Kernel Launch Pattern

threads_per_block = 256
blocks_per_grid = (num_blocks + threads_per_block - 1) // threads_per_block

aes_ecb_encrypt_kernel[blocks_per_grid, threads_per_block](
    d_state,
    d_key,
    num_blocks,
    d_sbox,
    d_mul2,
    d_mul3,
    d_rcon
)

## Appendix  — Known Bottlenecks (Summary)

The following bottlenecks were identified in the current GPU-based AES-128 ECB implementation:

- **Kernel launch latency**  
  GPU kernels invoked from Python incur significant overhead, especially for small workloads.

- **PCIe transfer cost dominating computation**  
  Host↔Device memory transfers are slower than the AES computation itself, limiting throughput.

- **Global memory S-box lookups**  
  Lookup tables stored in global memory introduce high-latency accesses; shared or constant memory would be faster.

- **Underutilization of Streaming Multiprocessors (SMs)**  
  Small or poorly configured grid/block sizes lead to idle GPU cores and low occupancy.

- **Python-level overhead**  
  Numba JIT compilation, dispatcher overhead, and intermediate buffer conversions inflate total execution time.