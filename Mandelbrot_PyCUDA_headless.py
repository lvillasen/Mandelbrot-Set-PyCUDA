import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import cm as cm
from pycuda.compiler import SourceModule

L = 400
N = 800
n_block = 16
n_grid = N // n_block
x0, y0 = -0.5, 0.0
side = 3.0
power = 2.0
i_cmap = 49
cmaps = [m for m in cm.datad if not m.endswith("_r")]

mod = SourceModule("""
#include <pycuda-complex.hpp>
typedef pycuda::complex<double> pyComplex;
__device__ float norma(pyComplex z) { return norm(z); }
__global__ void mandelbrot(double x0, double y0, double side, int L, double power, int *M) {
    int n_x = blockDim.x * gridDim.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int threadId = idy * n_x + idx;
    double delta = side / n_x;
    pyComplex c(x0 - side/2. + delta*idx, y0 - side/2. + delta*idy);
    pyComplex z = c;
    int h = 0;
    float R = 4.0;
    while (h < L && norma(z) < R) {
        z = pow(z, power) + c;
        h++;
    }
    M[threadId] = h;
}
""")

M = np.zeros((N, N), dtype=np.int32)
func = mod.get_function("mandelbrot")
func(np.float64(x0), np.float64(y0), np.float64(side),
     np.int32(L), np.float64(power), drv.Out(M),
     block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))

# Plot and save image
plt.figure(figsize=(8, 8))
plt.imshow(M, origin='lower', cmap=cmaps[i_cmap])
plt.title(f'Side={side:.2e}, x={x0:.2e}, y={y0:.2e}, {cmaps[i_cmap]}, L={L}')
plt.savefig("mandelbrot_cuda.png")
print("Imagen generada y guardada como mandelbrot_cuda.png")
