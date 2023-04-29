#include <iostream>
#include <Eigen/Dense>

#define IND(x, y, d) = ((int)((y * d) + x))

using namespace std;
using namespace Eigen;


template <typename T>
T* initVectorField(unsigned dim) {
    T *ret;
    cudaMalloc(&ret, dim*dim*sizeof(T));
    cudaMemset(ret, T::Zero(), dim*dim);
    return ret;
}


template <typename T>
T* initScalarField(unsigned dim) {
    T *ret;
    cudaMalloc(&ret, dim*dim*sizeof(T));
    cudaMemset(ret, (T)0, dim*dim);
    return ret;
}

/***
 * Bilinear Interpolation
 * https://en.wikipedia.org/wiki/Bilinear_interpolation
 */
__device__ Vector2f bilerp(Vector2f pos, Vector2f *field, unsigned dim) {
    int i = (int)pos(0);
    int j = (int)pos(1);
    double dx = (double)(pos(0) - i);
    double dy = (double)(pos(1) - j);

    if (i < 0 || i >= dim || j < 0 || j >= dim) {
        // Out of bounds.
        return Vector2f::Zero();
    } else {
        // Perform bilinear interpolation.
        Vector2f f00 = field[IND(x-1, y-1, dim)];
        Vector2f f01 = field[IND(x+1, y-1, dim)];
        Vector2f f10 = field[IND(x-1, y+1, dim)];
        Vector2f f11 = field[IND(x+1, y+1, dim)];
        Vector2f f0 = (1 - dx) * f00 + dx * f10;
        Vector2f f1 = (1 - dx) * f01 + dx * f11;
        return (1 - dy) * f0 + dy * f1;
    }
}

__device__ Vector2f next_poisson(float x, float b, float dx, unsigned dim) {
    // TODO: skeleton code; not tested
    int i = (int)pos(0);
    int j = (int)pos(1);
    float x_next = 0;
    x_next += (i - 1 < 0 || i - 1 >= dim || j < 0 || j >= dim) ? 0 : x[IND(i - 1, j, dim)];
    x_next += (i + 1 < 0 || i + 1 >= dim || j < 0 || j >= dim) ? 0 : x[IND(i + 1, j, dim)];
    x_next += (i < 0 || i >= dim || j - 1 < 0 || j - 1 >= dim) ? 0 : x[IND(i, j - 1, dim)];
    x_next += (i < 0 || i >= dim || j + 1 < 0 || j + 1 >= dim) ? 0 : x[IND(i, j + 1, dim)];
    x_next += (i < 0 || i >= dim || j < 0 || j >= dim) ? 0 : -dx*dx*b[IND(i, j, dim)];
    x_next /= 4;
}

__device__ Vector2f next_diffusion(Vector2f x, float dx,  float nu, float dt, unsigned dim) {
    // TODO: skeleton code; not tested
    int i = (int)pos(0);
    int j = (int)pos(1);
    Vector2f x_next = 0;
    float alpha = delta_x*delta_x/nu/dt;
    x_next += (i - 1 < 0 || i - 1 >= dim || j < 0 || j >= dim) ? 0 : x[IND(i - 1, j, dim)];
    x_next += (i + 1 < 0 || i + 1 >= dim || j < 0 || j >= dim) ? 0 : x[IND(i + 1, j, dim)];
    x_next += (i < 0 || i >= dim || j - 1 < 0 || j - 1 >= dim) ? 0 : x[IND(i, j - 1, dim)];
    x_next += (i < 0 || i >= dim || j + 1 < 0 || j + 1 >= dim) ? 0 : x[IND(i, j + 1, dim)];
    x_next += (i < 0 || i >= dim || j < 0 || j >= dim) ? 0 : -delta_x*delta_x*x[IND(i, j, dim)];
    x_next /= (4 + alpha);
}


/***
 * Computes the advection of the fluid.
 * 
 * x is the coordinate/position vector following notation of chp 38.
 * velfield is u, the velocity field as of the current time quanta.
 * field is the current field being updated.
*/
__device__ void advect(Vector2f x, Vector2f *field, Vector2f *velfield, float timestep, float rdx, unsigned dim) {
    Vector2f pos = x - timestep*rdx*velfield[IND(x(0), x(1), dim)];
    field[IND(x(0), x(1), dim)] = bilerp(pos, field, dim);
}


/***
 * Jacobi iteration for computing pressure and
 * viscous diffusion of fluid.
*/
template <typename T>
__device__ void jacobi(Vector2f x, T *field, float alpha, float beta, Vector2f b) {
    Vector2f f00 = field[IND(x - 1, y - 1, dim)];
    Vector2f f01 = field[IND(x + 1, y - 1, dim)];
    Vector2f f10 = field[IND(x - 1, y + 1, dim)];
    Vector2f f11 = field[IND(x + 1, y + 1, dim)];

    return (f00 + f01 + f10 + f11 + alpha*b) / beta;
}

__global__ void kernel(Vector2f *velocity, Vector2f *pressure) {
    Vector2f x(threadIdx.x, threadIdx.y);
    return;
}

int main(void) {
    // dimension of vector fields
    unsigned dim = 1024;
    // resolution of display
    unsigned res = 1024;
    // how many pixels a cell of the vector field represents
    float rdx = res / dim;
    
    Vector2f *dev_velocity = initVectorField<Vector2f>(dim);
    float *dev_pressure = initScalarField<float>(dim);

    // Iterate
    /*
    next_poisson(p, div_w, dx, dim);
    next_diffusion(u, dx, nu, dt, dim);
    */

    kernel<<<1, dim>>>();
    return 0;
}