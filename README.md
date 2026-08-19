# Weyl WebGPU Wave Simulator

A browser-based, GPU-accelerated numerical simulator for the Weyl equation using WebGPU.

The long-term goal of this project is to simulate the propagation of a two-component Weyl spinor in spatially and temporally varying vector potentials while maintaining interactive visualization and smooth frame rates on consumer GPUs.

The project is intentionally being developed incrementally so that every numerical, physical, and GPU-programming decision can be understood, tested, benchmarked, and documented.

---

## Project Goal

We want to numerically propagate a two-component Weyl spinor:

```math
\psi(\mathbf{x},t)
=
\begin{pmatrix}
\psi_+(\mathbf{x},t) \\
\psi_-(\mathbf{x},t)
\end{pmatrix}
```

under a Weyl Hamiltonian of the approximate form:

```math
H(t)
=
\boldsymbol{\sigma}
\cdot
\left(
-i\nabla
-
\mathbf{A}(\mathbf{x},t)
\right)
```

where:

* `σ = (σx, σy, σz)` are the Pauli matrices,
* `-i∇` is the momentum operator,
* `A(x,t)` is a spatially dependent vector potential,
* `A` may contain discontinuous or step-function spatial structures,
* those structures may move with time,
* and the wavefunction contains two complex spinor components at every spatial grid point.

This is ultimately a time-dependent quantum wave-propagation problem.

---

# Time-Dependent Vector Potential Strategy

The vector potential may move in time.

Rather than initially solving a continuously time-dependent Hamiltonian directly, the planned approach is to divide the evolution into intervals during which the Hamiltonian is treated as static.

For interval `j`:

```math
\mathbf{A}(\mathbf{x},t)
\approx
\mathbf{A}_j(\mathbf{x})
```

which defines a frozen Hamiltonian:

```math
H_j
=
\boldsymbol{\sigma}
\cdot
\left(
-i\nabla
-
\mathbf{A}_j(\mathbf{x})
\right)
```

The wavefunction is then propagated under this static Hamiltonian:

```math
\psi_{j+1}
=
e^{-iH_j\Delta t_j}\psi_j
```

The resulting wavefunction becomes the initial state for the next frozen Hamiltonian:

```math
H_j
\longrightarrow
H_{j+1}
```

The complete evolution therefore looks schematically like:

```math
\psi_0
\xrightarrow{H_0}
\psi_1
\xrightarrow{H_1}
\psi_2
\xrightarrow{H_2}
\psi_3
\longrightarrow
\cdots
```

The ordering is important because Hamiltonians associated with different positions of the vector potential generally do not commute:

```math
[H_j,H_{j+1}]
\neq
0
```

Therefore, each propagated wavefunction must be handed to the next Hamiltonian in chronological order.

---

# Planned Hamiltonian Decomposition

The kinetic part of the Weyl Hamiltonian becomes simple in momentum space.

Under the Fourier transform:

```math
-i\nabla
\longrightarrow
\mathbf{k}
```

Therefore, the kinetic portion can be evaluated as:

```math
F^\dagger
\left[
\boldsymbol{\sigma}
\cdot
\mathbf{k}
\right]
F
```

while the vector-potential contribution is local in position space:

```math
-
\boldsymbol{\sigma}
\cdot
\mathbf{A}(\mathbf{x})
```

The Hamiltonian can therefore be viewed schematically as:

```math
H
=
F^\dagger
\left[
\boldsymbol{\sigma}
\cdot
\mathbf{k}
\right]
F
-
\boldsymbol{\sigma}
\cdot
\mathbf{A}(\mathbf{x})
```

This suggests an implementation involving:

1. position-space spinor data,
2. forward FFT,
3. momentum-space action of the kinetic Weyl operator,
4. inverse FFT,
5. position-space action of the vector-potential term.

The exact implementation will be developed carefully.

In particular, the project should investigate how to minimize the number of FFT passes required for each application of the Hamiltonian.

The wavefunction should remain resident on the GPU whenever possible.

Unnecessary CPU-to-GPU and GPU-to-CPU transfers should be avoided.

---

# Chebyshev Time Propagation

For each frozen Hamiltonian, the intended propagation method is a Chebyshev polynomial expansion of the exponential time-evolution operator.

The quantity we want to compute is:

```math
e^{-iH\Delta t}\psi
```

The Hamiltonian must first be rescaled so that its spectrum lies approximately inside the interval:

```math
[-1,1]
```

Suppose the estimated lower and upper spectral bounds are:

```math
E_{\min}
\quad\text{and}\quad
E_{\max}
```

Define:

```math
a
=
\frac{E_{\max}-E_{\min}}{2}
```

and:

```math
c
=
\frac{E_{\max}+E_{\min}}{2}
```

Then define the rescaled Hamiltonian:

```math
\widetilde{H}
=
\frac{H-cI}{a}
```

The time-evolution operator can then be approximated through the Jacobi-Anger/Chebyshev expansion:

```math
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
\right]
```

where:

* `Jn` is a Bessel function of the first kind,
* `Tn` is a Chebyshev polynomial,
* and `N` is the chosen truncation order.

The expansion acts on the wavefunction:

```math
\psi(t+\Delta t)
\approx
e^{-ic\Delta t}
\left[
J_0(a\Delta t)\phi_0
+
2
\sum_{n=1}^{N}
(-i)^n
J_n(a\Delta t)\phi_n
\right]
```

where the Chebyshev vectors are generated through a three-term recurrence.

The first vector is:

```math
\phi_0
=
\psi
```

The second vector is:

```math
\phi_1
=
\widetilde H\psi
```

and subsequent vectors satisfy:

```math
\phi_{n+1}
=
2\widetilde H\phi_n
-
\phi_{n-1}
```

This recurrence is computationally important because the algorithm does not require explicitly constructing matrix powers such as:

```math
H^2,\ H^3,\ H^4,\ldots
```

Instead, only repeated applications of the Hamiltonian are required.

That is particularly attractive for GPU computing because the Hamiltonian can be implemented as an operation acting directly on GPU-resident wavefunction buffers.

---

# Current Spinor Memory Layout

The initial implementation uses 32-bit floating-point values.

A Weyl spinor contains two complex components:

```math
\psi
=
\begin{pmatrix}
\psi_+ \\
\psi_-
\end{pmatrix}
```

Each complex number contains a real and imaginary component.

The current memory layout for one spatial grid point is:

```text
[ Re(ψ+), Im(ψ+), Re(ψ−), Im(ψ−) ]
```

The JavaScript representation is a `Float32Array`.

For example:

```javascript
const psi = new Float32Array([1, 0, 0, 0]);
```

represents:

```math
\psi
=
\begin{pmatrix}
1+0i \\
0+0i
\end{pmatrix}
```

One `Float32` occupies four bytes.

Therefore, one Weyl spinor currently requires:

```math
4
\times
4\ \text{bytes}
=
16\ \text{bytes}
```

per spatial grid point.

This layout may later be reconsidered if a different organization provides better:

* memory coalescing,
* FFT compatibility,
* shader simplicity,
* cache behavior,
* or compute throughput.

No such optimization should be made without benchmarking.

---

# WebGPU Architecture

The simulation is being implemented with WebGPU rather than WebGL.

The basic WebGPU hierarchy currently understood and tested is:

```text
navigator.gpu
      ↓
GPUAdapter
      ↓
GPUDevice
      ↓
GPUBuffer
      ↓
Bind Groups
      ↓
Compute Pipelines
      ↓
Command Encoders
      ↓
GPU Queue
```

The browser currently reports an adapter with:

```text
vendor: NVIDIA
architecture: Lovelace
fallback adapter: false
```

The development GPU is an NVIDIA RTX 4070 Super.

A valid `GPUDevice` has successfully been created.

---

# Current WebGPU Storage Limit

The currently reported value of:

```javascript
device.limits.maxStorageBufferBindingSize
```

is:

```text
134217728 bytes
```

which is:

```math
128\ \text{MiB}
```

This value is **not the total amount of GPU VRAM**.

It is a WebGPU API limit describing the maximum amount of storage-buffer memory that a single shader buffer binding may expose under the current device configuration.

The distinction between:

```text
physical VRAM
```

and:

```text
WebGPU resource/binding limits
```

must remain explicit throughout development.

---

# Current Development Environment

Development is currently being performed with:

* Windows
* Visual Studio Code
* Live Server / Go Live
* Chrome / Chromium with WebGPU support
* JavaScript ES modules
* Git
* GitHub
* SSH authentication
* Ed25519 SSH key
* NVIDIA RTX 4070 Super
* NVIDIA Lovelace architecture

The repository uses the MIT License.

---

# Git and GitHub Configuration

The project is stored in a Git repository using the `main` branch.

The local repository is connected to GitHub through an SSH remote.

The development workflow currently follows:

```text
edit
  ↓
test locally
  ↓
git status
  ↓
git add
  ↓
git commit
  ↓
git push
```

SSH authentication has been confirmed successfully with GitHub.

The repository remote uses the SSH protocol rather than HTTPS authentication.

---

# Current Project State

The following functionality has been confirmed.

## Repository

* Git repository initialized
* `main` branch configured
* GitHub repository created
* GitHub remote configured
* SSH authentication working
* first commits pushed successfully
* MIT license added
* README documentation added

## Browser

* Live Server successfully serves the application
* JavaScript module successfully loads
* Chrome DevTools console is being used for debugging

## WebGPU

* `navigator.gpu` exists
* WebGPU returns a valid `GPUAdapter`
* NVIDIA adapter detected
* Lovelace architecture detected
* adapter is not a fallback adapter
* `GPUDevice` successfully created
* WebGPU device limits can be inspected

## Spinor Representation

One CPU-side Weyl spinor has been represented using:

```javascript
const psi = new Float32Array([1, 0, 0, 0]);
```

The browser reports:

```text
Float32Array(4)
byteLength: 16
```

which confirms the expected 16-byte layout.

## GPU Memory

A GPU storage buffer has been allocated for the spinor.

The current buffer has:

```text
size: 16 bytes
```

and was created with usage flags corresponding to:

```text
STORAGE
COPY_DST
```

The storage flag is intended to allow compute shaders to access the buffer.

The copy-destination flag is intended to allow data to be uploaded into the buffer.

---

# Exact Current Implementation Checkpoint

At the current development checkpoint we have:

```text
CPU
────────────────────────────

Float32Array

[1, 0, 0, 0]

16 bytes
```

and separately:

```text
GPU
────────────────────────────

GPUBuffer

16 bytes allocated
```

The CPU spinor has **not yet been transferred into the GPU buffer**.

Conceptually, the project is currently here:

```text
CPU Float32Array ψ
        │
        │   NOT DONE YET
        ▼
GPUBuffer ψ
```

The next intended WebGPU operation is:

```javascript
device.queue.writeBuffer(...)
```

which will perform the initial CPU-to-GPU upload.

That operation should be introduced as a small isolated development step and understood before proceeding further.

---

# Development Philosophy

This project is intentionally being implemented in very small interactive steps.

Each new WebGPU API call, mathematical operation, or data structure should be understood before moving forward.

The development loop should generally be:

```text
understand
    ↓
implement smallest piece
    ↓
observe
    ↓
test
    ↓
verify
    ↓
commit
    ↓
continue
```

Large blocks of untested implementation code should be avoided.

New code should include explanatory comments describing why each important line exists.

---

# Test-Driven Development Strategy

The project will use test-driven development wherever practical.

The intended development cycle is:

```text
write failing test
       ↓
implement smallest change
       ↓
make test pass
       ↓
refactor if needed
       ↓
run tests again
       ↓
commit
```

Testing will eventually be divided into several layers.

---

## Unit Tests

Unit tests will cover deterministic CPU-side mathematical functionality.

Examples include:

* complex-number arithmetic,
* spinor manipulation,
* Pauli matrix operations,
* wavefunction normalization,
* spatial-grid indexing,
* momentum-grid construction,
* vector-potential construction,
* spectral rescaling,
* Chebyshev recurrence,
* Bessel coefficients,
* Chebyshev coefficients,
* analytic reference solutions.

A likely unit-testing framework is Vitest.

The exact framework should be selected deliberately rather than treated as fixed prematurely.

---

## WebGPU Integration Tests

GPU-specific tests should verify operations including:

* GPU buffer creation,
* CPU-to-GPU transfer,
* GPU-to-CPU readback,
* WGSL shader compilation,
* compute-pipeline creation,
* GPU complex arithmetic,
* Pauli matrix application,
* FFT correctness,
* Hamiltonian application,
* GPU Chebyshev recurrence.

A browser automation framework such as Playwright may eventually be used so that these tests execute inside a real Chromium environment with WebGPU support.

---

## Numerical Physics Tests

Successful code execution is not sufficient evidence that the physics is correct.

The simulator should eventually contain numerical validation tests.

Important examples include:

### Norm Conservation

For unitary evolution:

```math
\langle\psi(t)|\psi(t)\rangle
=
\text{constant}
```

up to expected floating-point and truncation error.

---

### FFT Reconstruction

A forward transform followed by its corresponding inverse should reconstruct the original wavefunction:

```math
F^{-1}F\psi
\approx
\psi
```

within numerical tolerance.

---

### Free Weyl Propagation

With:

```math
\mathbf{A}=0
```

the numerical solution should be compared against analytically or independently computed free-particle evolution.

---

### Constant Vector Potential

For suitable constant values of:

```math
\mathbf{A}
```

the simulation should be compared against known momentum-shift behavior.

---

### Chebyshev Convergence

Increasing the Chebyshev truncation order should reduce propagation error until floating-point or other numerical errors dominate.

---

### Grid Convergence

Solutions should be compared while varying:

* spatial resolution,
* domain size,
* timestep,
* Chebyshev order.

---

### CPU vs GPU Reference Calculations

Small problem sizes should be computed both on the CPU and GPU.

The CPU implementation can act as a high-clarity reference implementation against which optimized GPU kernels are tested.

---

# Performance Goals

The project is intended to maintain smooth interactive visualization while numerical wave propagation is occurring.

Performance work should be measurement-driven rather than speculative.

Important principles include:

* keep wavefunction data on the GPU,
* minimize CPU-to-GPU transfers,
* minimize GPU-to-CPU readbacks,
* avoid unnecessary CPU/GPU synchronization,
* reuse GPU buffers,
* reuse compute pipelines,
* reuse bind groups where practical,
* batch GPU commands,
* minimize FFT passes,
* maintain GPU-friendly memory access,
* avoid unnecessary allocations during the simulation loop,
* benchmark workgroup sizes,
* separate simulation frequency from rendering frequency,
* profile before optimizing.

---

# Simulation Throughput vs Rendering FPS

The numerical solver and visual renderer represent different workloads.

Therefore, the project should distinguish between:

```text
simulation steps per second
```

and:

```text
rendered frames per second
```

A simulation does not necessarily need to perform exactly one propagation step for every rendered frame.

Eventually, the application may use a structure similar to:

```text
physics simulation loop
          ↓
GPU state
          ↓
render loop
```

where simulation work and visualization are coordinated without unnecessarily blocking one another.

---

# GPU Performance Philosophy

The objective is not simply to force the GPU to run at maximum electrical power.

Instead, good performance should come from efficient utilization.

Important concerns include:

```text
GPU occupancy
memory bandwidth
workgroup scheduling
buffer reuse
FFT efficiency
command submission
CPU/GPU synchronization
shader arithmetic
rendering workload
```

The application should avoid depending on hardware-specific assumptions unless profiling shows that they are justified.

---

# Planned Development Sequence

The expected development path is approximately:

```text
WebGPU initialization
        ↓
GPU buffer allocation
        ↓
CPU → GPU transfer
        ↓
GPU → CPU readback
        ↓
automated test infrastructure
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
kinetic Weyl operator
        ↓
vector potential A(x)
        ↓
complete Hψ operation
        ↓
spectral-bound estimation
        ↓
Hamiltonian rescaling
        ↓
Chebyshev recurrence
        ↓
Jacobi-Anger coefficients
        ↓
static-H propagation
        ↓
time-dependent / moving A(x,t)
        ↓
visualization
        ↓
profiling
        ↓
optimization
```

This sequence is a roadmap rather than an immutable specification.

Numerical or WebGPU constraints may require changes.

Any substantial architectural changes should be documented.

---

# Important Numerical Questions Still To Resolve

Several important choices remain intentionally unresolved.

These include:

* whether the initial simulator should be 1D, 2D, or 3D,
* the system of physical units,
* whether nondimensionalized variables should be used,
* spatial boundary conditions,
* FFT implementation strategy,
* FFT memory layout,
* grid dimensions,
* momentum-grid conventions,
* sign conventions for Fourier transforms,
* spectral-bound estimation,
* Chebyshev truncation criteria,
* timestep selection,
* moving-step representation,
* discontinuity handling,
* possible Gibbs phenomena,
* precision requirements,
* GPU buffer organization,
* visualization architecture,
* GPU timing methodology,
* performance benchmark methodology.

These decisions should not be silently assumed.

Each should be:

```text
discussed
   ↓
documented
   ↓
implemented
   ↓
tested
   ↓
validated
```

---

# Moving Step-Function Vector Potentials

One of the primary intended use cases is a vector potential containing spatial steps or discontinuities whose position changes with time.

A schematic example in one dimension might be:

```math
A(x,t)
=
A_0
\Theta
\left(
x-x_0(t)
\right)
```

where:

```math
\Theta(x)
```

denotes a step function and:

```math
x_0(t)
```

describes the moving interface.

The initial strategy is not to continuously mutate the Hamiltonian during a single Chebyshev expansion.

Instead:

```text
freeze A
   ↓
construct/use H_j
   ↓
propagate ψ
   ↓
move A
   ↓
construct/use H_(j+1)
   ↓
propagate new ψ
```

This approach should make the time-dependent problem easier to reason about and test before more sophisticated time-ordering schemes are considered.

---

# Long-Term Optimization Questions

Once correctness is established, several performance questions will become important.

Examples include:

* Should the two spinor components be interleaved or separated?
* Should complex values be stored as pairs of `f32` values?
* Can the Pauli-matrix action be fused into FFT-related kernels?
* Can vector-potential application be fused with another compute pass?
* How many FFTs are required for one Hamiltonian application?
* Can intermediate Chebyshev vectors reuse existing buffers?
* Can ping-pong buffers eliminate allocation?
* Can command encoders contain multiple Chebyshev iterations?
* When does command-buffer size become significant?
* What workgroup sizes perform best?
* Should rendering consume the simulation buffer directly?
* How frequently should diagnostic values be read back to the CPU?
* Can norm calculations be performed entirely on the GPU?
* How should reductions be implemented?
* What grid sizes preserve smooth rendering?
* What operations become memory-bandwidth limited?
* What operations become arithmetic-throughput limited?

These questions should eventually be answered through measurements.

---

# Repository Quality Goals

The repository is intended not only as an experiment but also as a professional software-engineering project.

The repository should progressively demonstrate:

* clear documentation,
* meaningful commit history,
* consistent commit messages,
* automated tests,
* numerical validation,
* modular architecture,
* documented design decisions,
* reproducible benchmarks,
* performance measurements,
* understandable code comments,
* issue tracking,
* versioned milestones,
* continuous integration where practical.

Commit messages should describe meaningful changes rather than vague statements such as:

```text
updated code
```

Preferred commit prefixes may include:

```text
feat:
fix:
test:
docs:
perf:
refactor:
chore:
```

For example:

```text
feat: upload initial Weyl spinor to GPU storage
```

or:

```text
test: verify Pauli sigma-x spinor transformation
```

---

# Guidance for Future LLM Sessions

If development continues in another AI conversation, provide the AI with this repository or, at minimum:

* this `README.md`,
* `index.html`,
* `main.js`,
* test files,
* shader files,
* configuration files.

The AI should preserve the following development style.

1. Work in very small interactive steps.
2. Introduce only a small number of lines of code at a time.
3. Add explanatory comments to newly introduced code.
4. Explain every new WebGPU concept.
5. Do not skip ahead to large implementations.
6. Introduce tests alongside mathematical functionality.
7. Prioritize correctness before optimization.
8. Keep GPU-performance considerations visible throughout development.
9. Do not silently assume FFT conventions.
10. Do not silently assume Weyl or Pauli-matrix conventions.
11. Do not silently assume Chebyshev scaling conventions.
12. Maintain a clear distinction between CPU memory and GPU memory.
13. Maintain a clear distinction between simulation throughput and rendering FPS.
14. Commit working checkpoints frequently.
15. Preserve numerical reference implementations when they are useful for testing optimized GPU code.

---

# Exact Handoff Point for a Future LLM

At the time this section was written, the program has successfully reached:

```text
navigator.gpu
      ↓
GPUAdapter
      ↓
GPUDevice
      ↓
GPUBuffer allocation
```

The CPU currently contains:

```javascript
const psi = new Float32Array([1, 0, 0, 0]);
```

representing:

```math
\psi
=
\begin{pmatrix}
1+0i \\
0+0i
\end{pmatrix}
```

The corresponding GPU buffer has already been allocated.

However, the actual transfer:

```text
CPU ψ
   ↓
GPU ψ
```

has not yet been implemented.

The next intended WebGPU API operation is:

```javascript
device.queue.writeBuffer(...)
```

The next AI assistant should **not jump ahead** to:

* WGSL shaders,
* FFTs,
* Pauli-matrix kernels,
* Hamiltonian propagation,
* Chebyshev recurrence,
* rendering.

The immediate next development step should be to understand and implement the smallest possible CPU-to-GPU transfer.

After that, the uploaded data should eventually be read back and verified before beginning more complicated GPU computation.

---

# License

This project is licensed under the MIT License.

See the repository's `LICENSE` file for details.
