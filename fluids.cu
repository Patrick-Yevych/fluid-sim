#include <iostream>
#include <math.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h> // for sleep function. use window.h for windows.
#endif
#include <eigen3/Eigen/Dense>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define IND(x, y, d) int((y) * (d) + (x))

using namespace std;
using Eigen::Vector2f;
using Eigen::Vector2d;

/***
 * Bilinear Interpolation
 * https://en.wikipedia.org/wiki/Bilinear_interpolation
 */
__device__ Vector2f bilerp(Vector2f pos, Vector2f* field, unsigned dim) {
    int i = pos(0);
    int j = pos(1);
    double dx = pos(0) - i;
    double dy = pos(1) - j;

    if (i < 0 || i >= dim || j < 0 || j >= dim) {
        // Out of bounds.
        return Vector2f::Zero();
    }
    else {
        // Perform bilinear interpolation.

        Vector2f f00 = (i - 1 < 0 || i - 1 >= dim || j - 1 < 0 || j - 1 >= dim) ? Vector2f::Zero() : field[IND(i - 1, j - 1, dim)];

        Vector2f f01 = (i + 1 < 0 || i + 1 >= dim || j - 1 < 0 || j - 1 >= dim) ? Vector2f::Zero() : field[IND(i + 1, j - 1, dim)];

        Vector2f f10 = (i - 1 < 0 || i - 1 >= dim || j + 1 < 0 || j + 1 >= dim) ? Vector2f::Zero() : field[IND(i - 1, j + 1, dim)];

        Vector2f f11 = (i + 1 < 0 || i + 1 >= dim || j + 1 < 0 || j + 1 >= dim) ? Vector2f::Zero() : field[IND(i + 1, j + 1, dim)];

        Vector2f f0 = (1 - dx) * f00 + dx * f10;
        Vector2f f1 = (1 - dx) * f01 + dx * f11;
        return (1 - dy) * f0 + dy * f1;
    }
}

__device__ float divergence(
    Vector2f x, Vector2f* from, float halfrdx, unsigned dim)
{
    int i = x(0);
    int j = x(1);
    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return Vector2f::Zero();

    Vector2f wL = (i - 1 < 0) ? Vector2f::Zero() : from[IND(i - 1, j, dim)];
    Vector2f wR = (i + 1 >= dim) ? Vector2f::Zero() : from[IND(i + 1, j, dim)];
    Vector2f wB = (j - 1 < 0) ? Vector2f::Zero() : from[IND(i, j - 1, dim)];
    Vector2f wT = (j + 1 <= dim) ? Vector2f::Zero() : from[IND(i, j + 1, dim)];

    return halfrdx * (wR(0) - wL(0), wT(1) - wB(1);
}


/***
 * only for computing gradient of p.
*/
__device__ Vector2f gradient(
    Vector2f x, float* p, float halfrdx, unsigned dim) {
    int i = x(0);
    int j = x(1);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return Vector2f::Zero();

    float pL = (i - 1 < 0)    ? 0 : p[IND(i - 1, j, dim)];
    float pR = (i + 1 >= dim) ? 0 : p[IND(i + 1, j, dim)];
    float pB = (j - 1 < 0)    ? 0 : p[IND(i, j - 1, dim)];
    float pT = (j + 1 >= dim) ? 0 : p[IND(i, j + 1, dim)];

    return halfrdx * Vector2f(pR - pL, pT - pB);
}


/***
 * Computes the advection of the fluid.
 *
 * x is the coordinate/position vector following notation of chp 38.
 * velfield is u, the velocity field as of the current time quanta.
 * field is the current field being updated.
*/
__device__ void advect(Vector2f x, Vector2f* field, Vector2f* velfield, float timestep, float rdx, unsigned dim) {
    Vector2f pos = x - timestep * rdx * velfield[IND(x(0), x(1), dim)];
    field[IND(x(0), x(1), dim)] = bilerp(pos, field, dim);
}

/***
 * Jacobi iteration for computing pressure and
 * viscous diffusion of fluid.
*/
template <typename T>
__device__ void jacobi(Vector2f x, T* field, float alpha, float beta, T b, T zero, unsigned dim) {
    int i = (int)x(0);
    int j = (int)x(1);

    T f00 = (i - 1 < 0 || i - 1 >= dim || j - 1 < 0 || j - 1 >= dim) ? zero : field[IND(i - 1, j - 1, dim)];

    T f01 = (i + 1 < 0 || i + 1 >= dim || j - 1 < 0 || j - 1 >= dim) ? zero : field[IND(i + 1, j - 1, dim)];

    T f10 = (i - 1 < 0 || i - 1 >= dim || j + 1 < 0 || j + 1 >= dim) ? zero : field[IND(i - 1, j + 1, dim)];

    T f11 = (i + 1 < 0 || i + 1 >= dim || j + 1 < 0 || j + 1 >= dim) ? zero : field[IND(i + 1, j + 1, dim)];

    field[IND(i, j, dim)] = (f00 + f01 + f10 + f11 + alpha * b) / beta;
}


__device__ void force(Vector2f x, Vector2f* field, Vector2f c, Vector2f F, float timestep, float r, unsigned dim) {
    float exp = (pow(x(0) - c(0), 2) + pow(x(1) - c(1), 2)) / 2;
    int i = x(0);
    int j = x(1);
    field[IND(i, j, dim)] = F * pow(timestep, exp);
}

__global__ void kernel(Vector2f* u, float* p, float rdx, float viscosity, Vector2f c, Vector2f F, int timestep, float r, unsigned dim)
{
    Vector2f x(threadIdx.x, threadIdx.y);

    //advection
    advect(x, u, u, timestep, rdx, dim);
    __syncthreads(); // barrier
    //diffusion
    float alpha = (rdx * rdx) / (viscosity * timestep);
    float beta = 4 + alpha;
    int i = x(0);
    int j = x(1);
    jacobi<Vector2f>(x, u, alpha, beta, u[IND(i, j, dim)], Vector2f::Zero(), dim);
    __syncthreads();

    //force application
    // apply force every 10 seconds
    if (timestep % 10 == 0)
        force(x, u, c, F, timestep, r, dim);
    __syncthreads();

    //pressure
    alpha = -1 * timestep * timestep;
    beta = 4;
    jacobi<float>(x, p, alpha, beta, divergence(x, u, (float)(rdx / 2), dim), 0, dim);
    __syncthreads();

    // u = w - nabla p
    u[IND(x(0), x(1), dim)] -= gradient(x, p, (float)(rdx / 2), dim);
    __syncthreads(); //potential redundant; implicit barrier between kernel calls
}

int main(void) {
    // quarter of second timestep
    float timestep = 0.25;
    // dimension of vector fields
    unsigned dim = 1024;
    // resolution of display
    unsigned res = 1024;
    // how many pixels a cell of the vector field represents
    float rdx = res / dim;

    // fluid parameters
    float viscosity = 1;

    // force parameters
    Vector2f c((int)(dim / 2), (int)(dim / 2));
    Vector2f F(1, 1);
    float r = 1;

    // half to alloc cpu/ram side u and p, then copy it to device/gpu side u, p.
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
    Vector2f *velocity = (Vector2f *)malloc(dim * dim * sizeof(Vector2f));
    float *pressure = (float *)malloc(dim * dim * sizeof(float));

    Vector2f *dev_velocity; // u
    cudaMalloc(&dev_velocity, dim * dim * sizeof(Vector2f));
    float *dev_pressure;
    cudaMalloc(&dev_pressure, dim * dim * sizeof(float));

    for (int i = 0; i < dim * dim; i++) {
        velocity[i] = Vector2f::Zero();
        pressure[i] = 0;
    }

    cudaMemcpy(dev_velocity, velocity, dim*dim*sizeof(Vector2f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pressure, pressure, dim * dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(dim, dim);
    while (true) {
        kernel << <1, threads >> > (dev_velocity, dev_pressure, rdx, viscosity, c, F, timestep, r, dim);
        sleep(timestep);
    }
    return 0;
}
