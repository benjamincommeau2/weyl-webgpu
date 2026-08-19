console.log("Weyl WebGPU project started");

document.querySelector("#status").textContent =
  "JavaScript loaded successfully.";

console.log("WebGPU available:", !!navigator.gpu); // Ask the browser whether the WebGPU API exists.

const adapter = await navigator.gpu.requestAdapter(); // Ask Chrome for the GPU adapter it has selected.

console.log("GPU adapter:", adapter); // Print the adapter so we can verify one was found.

console.log("GPU info:", adapter.info); // Show the GPU/vendor information Chrome exposes for this adapter.

const device = await adapter.requestDevice(); // Ask the selected adapter for a GPUDevice we can actually submit work to.

console.log("GPU device:", device); // Print the GPUDevice so we can confirm WebGPU created our usable compute device.

console.log("Max storage buffer bytes:", device.limits.maxStorageBufferBindingSize); // Show the largest storage buffer WebGPU lets one shader binding access.

const psi = new Float32Array([1, 0, 0, 0]); // Store one Weyl spinor as [Re(ψ+), Im(ψ+), Re(ψ−), Im(ψ−)].
console.log("One Weyl spinor:", psi); // Print the four float32 numbers so we can inspect our memory layout.

const psiBuffer = device.createBuffer({ size: psi.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); // Allocate GPU memory large enough for ψ; STORAGE lets shaders use it, COPY_DST lets us upload data into it.
console.log("GPU psi buffer:", psiBuffer); // Verify that WebGPU successfully created the GPU buffer.