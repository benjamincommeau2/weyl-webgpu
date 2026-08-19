# 2026/08/17

https://www.linkedin.com/feed/update/urn:li:activity:7495189884303925249/

WebGPU Dirac/Weyl Simulator — When the Fast Discretization Changes the Physics



I’ve been researching an interactive 3D Dirac/Weyl-family wave simulator in WebGPU.



The larger goal is to combine three mostly unrelated topics in one simulation: 3D wave mechanics, Deutsch closed-timelike-curve (D-CTC) quantum computation, and Smith-chart-style reflection/transmission analysis. Right now I’m building the wave/scattering foundation.



The current two-component model is



H = σ · (p − A(x))



where A(x) acts as an effective vector potential.



For a Weyl state, spinor orientation is tied to kinetic momentum q = p − A. If A changes across an interface, the allowed transmitted momentum and spinor change. Matching can require a reflected component; in other regimes the transmitted normal momentum becomes imaginary and the mode is evanescent. Special matching can still produce Klein-like perfect transmission. [2]



Could A simply be removed by a position-dependent phase transformation? Only if it is pure gauge:



A = ∇χ.



A gauge transformation can add or subtract a gradient, but cannot change ∇ × A. An interface configuration with nonzero curl therefore cannot be transformed away.



Numerically, my first approach looked ideal for WebGPU: centered finite differences plus matrix-free Chebyshev propagation.



After scaling H → H̃ so its spectrum lies in [−1,1],

U(Δt) ≈ exp(−icΔt)[J₀(aΔt) + 2 Σₙ₌₁ᴷ (−i)ⁿ Jₙ(aΔt)Tₙ(H̃)].



The recurrence reduces evolution to repeated Hψ operations instead of constructing or diagonalizing a huge Hamiltonian. [1]



The problem appeared in the spatial derivative. Centered differences replace

k → sin(kΔx)/Δx,



which changes the Weyl dispersion and creates extra low-energy lattice modes: the fermion-doubling problem. [3]



For scattering, this matters because sharp interfaces contain high spatial frequencies that can couple into lattice modes absent from the continuum problem.



I’m now investigating Fourier-pseudospectral differentiation while keeping Chebyshev propagation:



Hψ = F⁻¹[(σ · k)Fψ] − σ · A(x)ψ.



FFT-based pseudospectral methods are established for time-dependent Dirac equations [4], but on a 256³ WebGPU grid this replaces a cheap local stencil with repeated 3D FFTs.



The question I’m testing now is whether improved scattering fidelity justifies the added GPU memory traffic.



Eventually, those reflection/transmission amplitudes are what I want to connect to the Smith-chart part of the larger experiment.



References:



[1] Tal-Ezer & Kosloff, JCP 81, 3967 (1984), DOI 10.1063/1.448136

[2] Erementchouk & Mazumder, PLA 381, 2866 (2017), DOI 10.1016/j.physleta.2017.06.055

[3] Kaplan & Sen, PRL 132, 141604 (2024), DOI 10.1103/PhysRevLett.132.141604

[4] Antoine & Lorin, JCP 395, 583 (2019), DOI 10.1016/j.jcp.2019.06.020

#ScientificComputing #WebGPU #ComputationalPhysics

# 2026/08/19

https://www.linkedin.com/feed/update/urn:li:share:7495863937259966464/

WebGPU Weyl Simulator: Moving From Numerical Design to a Reproducible GPU Project



I’ve made the repository for my WebGPU Weyl simulator public and started turning the numerical design from my last post into a testable implementation.



The repo is here:

https://lnkd.in/e-p2rRnC



The current target remains a two-component Weyl equation,



H(t) = σ · (-i∇ - A(x,t)),



with moving step-function vector potentials.



For the time dependence, I’m starting with a frozen-Hamiltonian approach. During each short interval, A(x,t) is treated as static, the state is propagated under that Hamiltonian, and the resulting wavefunction is handed to the next Hamiltonian after the step moves.



The planned propagation is still Chebyshev/Jacobi-Anger:



exp(-iHΔt)ψ



with repeated applications of H through the Chebyshev recurrence.



For the spatial derivative, I’m continuing with the Fourier-pseudospectral direction from my previous post:



Hψ = F⁻¹[(σ · k)Fψ] - σ · A(x)ψ.



I’m building the implementation in deliberately small checkpoints so I can test each numerical and GPU assumption before stacking more complexity on top.



So far I have:



• set up the public GitHub repository with SSH authentication and an MIT license

• documented the mathematical model, numerical roadmap, performance goals, and exact development handoff point in the README

• initialized WebGPU in Chrome

• confirmed an NVIDIA Lovelace adapter and GPUDevice

• inspected WebGPU storage-buffer limits

• represented one complex two-component Weyl spinor as four Float32 values

• allocated the first GPU storage buffer for that spinor



The next step is intentionally small: copy that 16-byte spinor from CPU memory into the GPU buffer, read it back, and verify the result before writing the first WGSL compute shader.



I’m also adding a test-driven development workflow as the project grows. I want CPU reference tests, GPU integration tests, and physics-level validation such as norm conservation, FFT reconstruction, and Chebyshev convergence.



Once those foundations are trustworthy, I’ll start building toward the FFT Hamiltonian application and then the moving-step propagation.



AI-assisted drafting was used for this post. I reviewed the technical content and wording before publishing.



#WebGPU #ScientificComputing #ComputationalPhysics #OpenSource