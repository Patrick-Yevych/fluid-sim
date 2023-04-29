#include <iostream>
#include <Eigen/Dense>

#define IND(x, y, d) = ((int)((y * d) + x))

using namespace std;
using namespace Eigen;

__device__ Vector2f* initializeField(unsigned dim) {
    Vector2f *ret;
    cudaMalloc(&ret, dim*dim);
    for (int i = 0; i < dim*dim; i++) ret[i] << 0, 0;
    return ret;
}

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

__device__ void advect(Vector2f x, Vector2f *field, Vector2f *velfield, float timestep, float rdx, unsigned dim) {
    Vector2f pos = x - timestep*rdx*velfield[IND(x(0), x(1), dim)];
    field[IND(x(0), x(1), dim)] = bilerp(pos, field, dim);
}

int main(void) {
    return 0;
}