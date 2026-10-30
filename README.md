# CudaColorUnmixing

A CUDA port of the soft color segmentation method presented in:

> Yağiz Aksoy, Tunç Ozan Aydin, Aljoša Smolić, and Marc Pollefeys.
> Unmixing-Based Soft Color Segmentation for Image Manipulation.
> ACM Trans. Graph. 36, 2, Article 19 (March 2017).
> DOI: https://doi.org/10.1145/3002176

The original CPU implementation this port is based on is
[ColorUnmixing](https://github.com/yaksoy/ColorUnmixing) by Yağiz Aksoy.
This project re-implements the same pipeline with a CUDA backend and a
backend-agnostic interface, so the CPU and CUDA paths can be compared
on the same inputs.

## What it does

Given a 32-bit BGRA BMP, the pipeline decomposes the image into a set of
soft color segments ("layers") with per-pixel alpha channels. Each layer
contains:

- `layer_XX_color.bmp`   - flat color layer (RGB)
- `layer_XX_alpha.bmp`   - per-pixel alpha
- `layer_XX_overlay.bmp` - Photoshop-style overlay layer
- `refined_XX_*.bmp`     - same as above after matte refinement

Overlaying all layers reproduces the original image (alpha-add
compositing, see the paper for details).

## Architecture

```
src/common/    CPU backend, host operations, I/O, solver, pipeline
src/cuda/      CUDA backend, device kernels, warp-level batch solver
tests/         cpu_smoke, cuda_smoke, bmp_smoke, l2_verify, l3_verify, l4_verify
color-unmixing/ Reference CPU implementation from the original authors
```

The `IBackend` interface in `src/common/backend.hpp` dispatches every
stage (luminance, gradient, guided filter, normal distribution,
per-pixel solve) to either the CPU or CUDA implementation. The pipeline
in `src/common/pipeline.cpp` is backend-agnostic.

The batch solver (`src/cuda/batch_solver.cu`) assigns one warp per pixel
and runs L-BFGS + ALM entirely on the GPU. L-BFGS history, gradient
buffers and the augmented Lagrangian state live in shared memory.

## Requirements

- CMake 3.25 or newer
- Visual Studio 2022 with the C++ toolset
- CUDA Toolkit 12.6 (tested)
- Eigen 3 (via vcpkg)
- NLopt (only for the reference `color-unmixing` target)
- vcpkg with `x64-windows-static` triplet

## Build

```powershell
.\build.ps1
```

The preset `cuda` builds `unmixing_common`, `unmixing_cuda`, all
smoke tests and the verification binaries. Presets `cpu`, `cuda`,
`vulkan`, `all` are defined in `CMakePresets.json`.

CUDA architectures are currently `52;61` in `CMakeLists.txt`. Adjust
`CMAKE_CUDA_ARCHITECTURES` if you target different GPUs.

## Usage

Synthetic gradient image (no input file required):

```powershell
.\build\cuda\Release\cuda_smoke.exe 64 64 out_cuda
.\build\cuda\Release\cpu_smoke.exe  64 64 out
```

Real image with backend selection:

```powershell
.\build\cuda\Release\bmp_smoke.exe leaves.bmp out_cuda cuda
.\build\cuda\Release\bmp_smoke.exe leaves.bmp out_cpu  cpu
```

The original reference implementation is available as:

```powershell
.\build\cuda\color-unmixing\Release\color-unmixing.exe leaves.bmp out_ref
```

## Verification

Three verification levels are provided:

| Binary | What it checks |
|--------|----------------|
| `l2_verify` | Per-pixel CPU solver vs an independent CPU reference |
| `l3_verify` | Per-pixel CPU solver vs CUDA batch solver, 1000 random pixels |
| `l4_verify` | Full pipeline CPU vs CUDA on a real image |

See `docs/VALIDATION.md` for the current results and the reason the
pixel-level checks do not pass with a strict tolerance.

## Known limitations

The CPU and CUDA backends produce structurally identical decompositions
(same number of layers, same seed sequence), but differ at the pixel
level for approximately 1-5% of pixels. Root cause: the L-BFGS + ALM
optimization is non-convex, and the floating-point reduction order
differs between Eigen (sequential) and the warp-level CUDA reduction
(butterfly). Once the iterates diverge by ~1e-16, the outer ALM loop
amplifies the difference and some pixels converge to a different local
minimum.

This is not a bug in the CUDA port. The independent CPU reference in
`l2_verify` reproduces the same multi-solution behavior on the CPU side
alone. See `docs/VALIDATION.md` for data.

If bit-exact CPU/CUDA output is required, replace the warp reduction in
`batch_solver.cu` with a sequential reduction and rerun the tests. This
costs roughly 10-30x performance and does not guarantee identical
results because the outer ALM loop can still diverge.

## References

- Original paper: https://doi.org/10.1145/3002176
- Original CPU implementation: https://github.com/yaksoy/ColorUnmixing
- Green-screen keying via color unmixing (predecessor formulation):
  https://doi.org/10.1145/2980179.2980235

## License

MIT. See `LICENSE`. The reference implementation under
`color-unmixing/` retains its original MIT license from the upstream
repository.