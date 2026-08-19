# Weyl WebGPU Wave Simulator

A browser-based, GPU-accelerated numerical simulator for the Weyl equation using WebGPU.

The long-term goal of this project is to simulate the propagation of a two-component Weyl spinor in spatially and temporally varying vector potentials, while maintaining interactive visualization and smooth frame rates on consumer GPUs.

The project is intentionally being developed incrementally so that every numerical, physical, and GPU programming decision can be understood, tested, and benchmarked.

---

## Project Goal

We want to numerically propagate a two-component Weyl spinor

\[
\psi(\mathbf{x},t)
==================

\begin{pmatrix}
\psi_+(\mathbf{x},t) \
\psi_-(\mathbf{x},t)
\end{pmatrix}
\]

under a Weyl Hamiltonian of the approximate form

[
H(t)
====

\sigma \cdot \left(-i\nabla - A(\mathbf{x},t)\right),
]

where:

* (\sigma = (\sigma_x,\sigma_y,\sigma_z)) are the Pauli matrices,
* (-i\nabla) is the momentum operator,
* (A(\mathbf{x},t)) is a spatially dependent vector potential,
* (A) may contain discontinuous or step-function spatial structures,
* and those structures may move with time.

The wavefunction is complex-valued and has two spinor components at every spatial grid point.

---

## Time-Dependent Vector Potential Strategy

The vector potential may move in time.

Rather than initially solving the fully continuously time-dependent Hamiltonian directly, the planned approach is to divide evolution into intervals during which the Hamiltonian is treated as static.

For interval (j),

[
A(\mathbf{x},t)
\approx
A_j(\mathbf{x}),
]

giving a frozen Hamiltonian

[
H_j
===

\sigma\cdot
\left(-i\nabla-A_j(\mathbf{x})\right).
]

The state is propagated under this static Hamiltonian:

[
\psi_j
\longrightarrow
\psi_{j+1}
==========

e^{-iH_j\Delta t_j}\psi_j.
]

The resulting wavefunction is then handed directly to the next Hamiltonian:

[
H_j
\rightarrow
H_{j+1}.
]

Thus the simulation follows the ordered sequence

[
\psi_0
\xrightarrow{H_0}
\psi_1
\xrightarrow{H_1}
\psi_2
\xrightarrow{H_2}
\cdots.
]

The ordering is important because Hamiltonians corresponding to different positions of the vector potential generally do not commute.

---

## Planned Hamiltonian Decomposition

The kinetic part of the Hamiltonian becomes simple in momentum space.

Using the Fourier transform,

[
-i\nabla
\longrightarrow
\mathbf{k}.
]

The Hamiltonian can therefore be viewed schematically as

[
H
=

F^\dagger
\left[
\sigma\cdot\mathbf{k}
\right]
F
-

\sigma\cdot A(\mathbf{x}).
]

This suggests an implementation involving:

1. position-space spinor data,
2. forward FFT,
3. momentum-space action of (\sigma\cdot\mathbf{k}),
4. inverse FFT,
5. position-space action of (-\sigma\cdot A).

The exact number of FFT operations and the most efficient implementation strategy will be investigated carefully.

The project should avoid unnecessary CPU/GPU transfers and should keep the wavefunction resident on the GPU whenever possible.

---

## Chebyshev Time Propagation

For each frozen Hamiltonian, the intended propagation method is a Chebyshev polynomial expansion of the exponential operator.

We want to approximate

[
e^{-iH\Delta t}\psi.
]

The Hamiltonian must first be rescaled so that its spectrum lies approximately inside

[
[-1,1].
]

Define

[
\widetilde H
============

\frac{H-cI}{a},
]

where (a) and (c) are derived from estimates of the lower and upper spectral bounds.

The propagator can then be represented using the Jacobi-Anger/Chebyshev expansion

[
e^{-iH\Delta t}
\approx
e^{-ic\Delta t}
\left[
J_0(a\Delta t)T_0(\widetilde H)
+
2
\sum_{n=1}^{N}
(-i)^n
J_n(a\Delta t)
T_n(\widetilde H)
\right].
]

The Chebyshev vectors are generated through the three-term recurrence

[
\phi_0=\psi,
]

[
\phi_1=\widetilde H\psi,
]

and

[
\phi_{n+1}
==========

2\widetilde H\phi_n-\phi_{n-1}.
]

This is important computationally because the algorithm does not require explicitly calculating large matrix powers.

Only repeated applications of the Hamiltonian are required.

---

## Current Spinor Memory Layout

The initial implementation uses 32-bit floating-point values.

One Weyl spinor is represented as

```text
[ Re(ψ+), Im(ψ+), Re(ψ−), Im(ψ−) ]
```

using a JavaScript `Float32Array`.

For example,

[
\psi
====

\begin{pmatrix}
1\
0
\end{pmatrix}
]

is represented by

```javascript
new Float32Array([1, 0, 0, 0]);
```

Each grid point therefore currently requires

```text
4 floats × 4 bytes = 16 bytes
```

for the wavefunction itself.

This layout may later be reconsidered if another structure gives better memory coalescing or simplifies FFT operations.

---

## WebGPU Architecture

The simulation is being implemented using WebGPU rather than WebGL.

The basic WebGPU hierarchy currently understood and tested is

```text
navigator.gpu
      ↓
GPUAdapter
      ↓
GPUDevice
      ↓
GPUBuffer / shaders / pipelines / command encoders
```

The browser currently reports:

```text
vendor: NVIDIA
architecture: Lovelace
fallback adapter: false
```

The development GPU is an NVIDIA RTX 4070 Super.

The currently reported maximum storage-buffer binding size is

```text
134217728 bytes
```

or

```text
128 MiB
```

per storage-buffer binding.

This does not represent total GPU VRAM. It is a WebGPU binding limit exposed by the device.

---

## Current Development Environment

Development is currently being performed with:

* Windows
* Visual Studio Code
* Live Server / Go Live
* Chrome or another Chromium browser with WebGPU support
* Git
* GitHub
* SSH authentication using an Ed25519 key
* NVIDIA Lovelace GPU

The repository uses the MIT License.

---

## Current Project State

The following functionality has been confirmed:

* Git repository initialized
* GitHub remote configured
* SSH authentication with GitHub working
* `main` branch pushed to GitHub
* MIT license added
* Live Server successfully serves the application
* JavaScript module successfully loads
* `navigator.gpu` exists
* WebGPU returns a valid `GPUAdapter`
* NVIDIA Lovelace adapter detected
* adapter is not a fallback adapter
* `GPUDevice` successfully created
* WebGPU device limits can be inspected
* one CPU-side Weyl spinor has been represented with `Float32Array`
* one GPU storage buffer has been allocated for that spinor

At the present development checkpoint, the CPU spinor data has **not yet been copied into the GPU buffer**.

That is intentionally the next implementation step.

---

## Current CPU Data

The current CPU-side test spinor is

```javascript
const psi = new Float32Array([1, 0, 0, 0]);
```

representing

[
\psi=
\begin{pmatrix}
1+0i\
0+0i
\end{pmatrix}.
]

A corresponding 16-byte GPU storage buffer has been created using WebGPU.

The next immediate goal is to transfer those 16 bytes from CPU memory to GPU memory.

---

## Development Philosophy

This project is intentionally being implemented in very small steps.

Each new API or mathematical operation should be understood before proceeding.

The development loop should generally be

```text
understand
    ↓
implement smallest piece
    ↓
test
    ↓
inspect result
    ↓
commit
    ↓
continue
```

Large blocks of untested code should be avoided.

---

## Test-Driven Development Strategy

The project will use test-driven development where practical.

The intended progression is:

```text
write failing test
      ↓
implement smallest amount of code
      ↓
make test pass
      ↓
refactor
      ↓
commit
```

Testing will eventually be divided into several layers.

### Unit Tests

Unit tests will cover deterministic CPU-side mathematical operations such as:

* complex arithmetic,
* spinor manipulation,
* Pauli matrix operations,
* normalization,
* grid indexing,
* momentum-grid construction,
* spectral rescaling,
* Chebyshev recurrence,
* Bessel/Chebyshev coefficients,
* vector potential construction.

These tests should run quickly without requiring a GPU.

A likely unit-testing framework is Vitest.

### WebGPU Integration Tests

GPU-specific tests will verify operations such as:

* GPU buffer creation,
* CPU-to-GPU transfer,
* GPU-to-CPU readback,
* WGSL shader correctness,
* GPU complex arithmetic,
* Pauli matrix application,
* FFT correctness,
* Hamiltonian application,
* Chebyshev recurrence performed on the GPU.

A browser automation framework such as Playwright may eventually be used so that tests execute inside a real browser with WebGPU support.

### Numerical Physics Tests

The simulator should also include physics-level validation.

Examples include:

* norm conservation for unitary propagation,
* comparison with analytically solvable cases,
* free Weyl propagation,
* constant vector potential cases,
* convergence as Chebyshev order increases,
* convergence as the timestep decreases,
* FFT forward/inverse reconstruction,
* comparison of CPU reference calculations against GPU results.

These tests are especially important because code can execute correctly while still implementing incorrect physics.

---

## Performance Goals

The project is intended to maintain smooth interactive visualization while numerical propagation is occurring.

Performance work should be evidence-driven rather than speculative.

Important principles include:

* keep wavefunction data on the GPU,
* minimize CPU-to-GPU transfers,
* minimize GPU-to-CPU readbacks,
* avoid synchronizing the CPU and GPU unnecessarily,
* reuse GPU buffers instead of repeatedly allocating them,
* reuse compute pipelines,
* reuse bind groups when practical,
* batch GPU commands,
* minimize FFT passes,
* maintain contiguous/coalesced memory access,
* choose workgroup sizes through benchmarking,
* distinguish simulation timestep from rendering framerate,
* profile before optimizing.

The simulator should eventually measure both:

```text
simulation steps / second
```

and

```text
rendered frames / second
```

because these represent different performance characteristics.

---

## Planned Development Sequence

The expected development path is approximately:

```text
WebGPU initialization
        ↓
GPU buffers
        ↓
CPU → GPU transfer
        ↓
GPU → CPU verification
        ↓
first WGSL compute shader
        ↓
complex arithmetic
        ↓
Weyl spinor operations
        ↓
Pauli matrices
        ↓
spatial grid
        ↓
momentum grid
        ↓
FFT
        ↓
kinetic Hamiltonian
        ↓
vector potential A(x)
        ↓
complete Hψ operation
        ↓
spectral bounds
        ↓
Chebyshev recurrence
        ↓
Jacobi-Anger coefficients
        ↓
static-H propagation
        ↓
moving A(x,t)
        ↓
visualization
        ↓
profiling
        ↓
optimization
```

This sequence may change as numerical and WebGPU constraints become better understood.

---

## Important Numerical Questions Still To Resolve

The following design questions remain intentionally open:

* initial dimensionality: 1D, 2D, or 3D,
* physical units and nondimensionalization,
* spatial boundary conditions,
* FFT implementation,
* FFT data layout,
* grid dimensions,
* spectral bound estimation,
* Chebyshev truncation criterion,
* timestep selection,
* representation of moving step potentials,
* handling discontinuities and possible Gibbs phenomena,
* whether single precision is sufficient for all stages,
* GPU buffer organization,
* visualization method,
* GPU timing/profiling methodology.

These decisions should not be silently assumed.

They should be discussed, implemented, tested, and documented.

---

## Guidance for Future LLM Sessions

If development continues in another AI conversation, provide the AI with this repository or at minimum this README together with the current source files.

The AI should preserve the following development style:

1. Work in very small interactive steps.
2. Add only a few lines of code at a time.
3. Put explanatory comments on newly introduced code.
4. Explain what every new WebGPU concept means.
5. Do not skip ahead to large implementations.
6. Introduce tests alongside new mathematical functionality.
7. Prioritize numerical correctness before optimization.
8. Keep GPU performance considerations visible throughout the design.
9. Do not assume FFT, Chebyshev, or Weyl conventions without explicitly documenting them.
10. Maintain the distinction between CPU memory and GPU memory.
11. Maintain the distinction between simulation throughput and rendering FPS.
12. Commit working checkpoints frequently.

### Exact Current Handoff Point

The project currently has:

```text
CPU Float32Array ψ
        ↓
GPUBuffer allocated
```

but has not yet performed

```text
CPU Float32Array ψ
        ↓
device.queue.writeBuffer(...)
        ↓
GPUBuffer containing ψ
```

The next coding step should therefore be the smallest possible CPU-to-GPU transfer using `GPUQueue.writeBuffer`.

Do not jump directly to shaders, FFTs, or the Chebyshev propagator before verifying that transfer and eventually reading the data back correctly.
