# Q4_F16: RDNA-Native 4-bit Quantization Format

## Motivation

Existing GGUF quantization formats (Q4_K, Q6_K, etc.) use complex metadata encoding
optimized for NVIDIA's dp4a instruction path. On RDNA1 (gfx1010), the dequant overhead
from Q4_K's 6-bit packed scale encoding and manual half-to-float conversion consumes
~10-12 ALU instructions per element, limiting the Q4_K GEMV to 178.8 GB/s (39.9% of
448 GB/s peak bandwidth).

Q4_F16 stores scales as native FP16 values, reducing dequantization to a single FP16
FMA instruction. This plays to RDNA's strength: native `_Float16` arithmetic at 2x
FP32 rate across all generations (RDNA1-RDNA4).

## Format Variants

### Q4_F16_G64 (Primary)

36 bytes per 64 elements = 0.5625 bytes/weight (matches Q4_K density).

```
Offset  Size  Field
0       2     _Float16 scale   — group scale factor
2       2     _Float16 min     — group minimum (negative bias baked in, added to dequant)
4       32    uint8[32]        — 64 x 4-bit values, packed 2 per byte (low nibble first)
---
Total: 36 bytes per 64 elements
```

**Nibble packing:** Byte `quants[i]` contains:
- Element `i` in bits [3:0] (low nibble)
- Element `i+32` in bits [7:4] (high nibble)

This maps to a 32-thread warp: thread `tid` reads `quants[tid]`, extracts both nibbles,
and processes elements `tid` (low) and `tid+32` (high).

**Quantization range:** Each nibble stores values 0-15. The dequantized weight is:
```
weight = nibble * scale + min
```
where `min` is typically negative (representing the minimum weight in the group) and
`scale = (max_weight - min_weight) / 15.0`.

### Q4_F16_G32 (Variant)

20 bytes per 32 elements = 0.625 bytes/weight.

```
Offset  Size  Field
0       2     _Float16 scale
2       2     _Float16 min
4       16    uint8[16]        — 32 x 4-bit values, packed 2 per byte
---
Total: 20 bytes per 32 elements
```

**Nibble packing:** Byte `quants[i]` contains:
- Element `i` in bits [3:0] (low nibble, threads 0-15)
- Element `i+16` in bits [7:4] (high nibble, threads 16-31)

Finer granularity (32 vs 64 elements per group) yields better quantization quality
at 11% higher storage cost.

## Dequantization

Both variants use identical dequant logic:

```c
// HIP kernel inner loop (per element):
unsigned int nibble = (tid < half) ? (qbyte & 0xF) : (qbyte >> 4);
_Float16 w = (_Float16)((unsigned short)nibble) * scale + min;  // v_fma_f16
float w32 = (float)w;                                           // v_cvt_f32_f16
sum += w32 * x[k];                                              // v_fma_f32
```

**Instruction count per element: ~5 ALU**
1. `v_and_b32` or `v_lshrrev_b32` — nibble extraction
2. `v_cvt_f16_u16` — integer to FP16
3. `v_fma_f16` — dequant FMA (runs at 2x FP32 rate on RDNA)
4. `v_cvt_f32_f16` — widen to FP32
5. `v_fma_f32` — accumulate

Compare Q4_K: ~10-12 ALU per element (6-bit scale unpacking + manual half_to_float).

## GEMV Kernel Design

- **32 threads per block** (single RDNA warp)
- **`__launch_bounds__(32, 20)`** — target 20 blocks/CU for max occupancy
- **Warp shuffle reduction** — no shared memory needed
- **FP16 dequant, FP32 accumulate** — native FP16 for speed, FP32 for dot product precision

### G64 kernel loop (2 elements per thread per block)
```
for each block in row:
    load scale, min (f16, broadcast across warp)
    load quants[tid] (1 byte per thread)
    dequant low nibble → FMA into sum with x[block*64 + tid]
    dequant high nibble → FMA into sum with x[block*64 + 32 + tid]
warp_shuffle_reduce(sum)
```

### G32 kernel loop (1 element per thread per block)
```
for each block in row:
    load scale, min (f16, broadcast)
    load quants[tid & 15] (shared byte, 2:1 thread ratio)
    extract nibble based on tid < 16
    dequant → FMA into sum with x[block*32 + tid]
warp_shuffle_reduce(sum)
```

## Expected Performance (gfx1010 / RX 5700 XT)

| Metric | Q4_K (current) | Q4_F16_G64 (target) |
|--------|----------------|---------------------|
| ALU per element | 10-12 | ~5 |
| Bytes/weight | 0.5625 | 0.5625 |
| GEMV bandwidth | 178.8 GB/s (39.9%) | 280-320 GB/s (62-71%) |
| TinyLlama 1.1B | 106.1 tok/s | 150-190 tok/s |

## Conversion from Q4_K

A Q4_K super-block (144 bytes, 256 elements) converts to exactly 4 Q4_F16_G64 blocks
(4 x 36 = 144 bytes), preserving file size:

1. Decode the 6-bit packed scales/mins from Q4_K metadata
2. Pre-multiply: `effective_scale = d * sub_block_scale_int`
3. Pre-multiply: `effective_min = -(dmin * sub_block_min_int)`
4. Store as FP16 scale and min per G64 block
5. Repack nibbles: Q4_K interleaves sub-blocks; Q4_F16 stores sequentially

## Portability

Q4_F16 is hardware-agnostic in format. Only the GEMV kernel changes per architecture:

| RDNA Gen | GFX ID | Kernel Strategy |
|----------|--------|-----------------|
| RDNA1 | gfx1010 | `v_fma_f16` scalar FP16 |
| RDNA2 | gfx1030 | `v_fma_f16` or dp4a hybrid |
| RDNA3 | gfx1100 | WMMA wave matrix ops |
| RDNA4 | gfx1200 | WMMA / BF16 extensions |
