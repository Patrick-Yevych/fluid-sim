#include "const.h"

/**
 * Perform Jacobi iteration for the diffusion vector
 * @param x is the coordinate/position vector following notation of paper.
 * @param field The relevant vector field
 * @param rdx Reciprocal of the grid scale
 * @param visc Viscosity of the fluid
 * @param dt Timestep
 * @param dim The maximum dimension of the field [for bound checking]
 * @authors Samaria Mulligan, Patrick Yevych
 */
__device__ void next_diffusion(Vector2f x, Vector2f *field, float rdx, float visc, float dt, unsigned dim) {
    int i = (int)x(0);
    int j = (int)x(1);
    Vector2f x_next = Vector2f::Zero();
    float alpha = rdx * rdx / (visc * dt);
    x_next += (i - 1 < 0 || i - 1 >= dim || j < 0 || j >= dim) ? Vector2f::Zero() : field[IND(i - 1, j, dim)];
    x_next += (i + 1 < 0 || i + 1 >= dim || j < 0 || j >= dim) ? Vector2f::Zero() : field[IND(i + 1, j, dim)];
    x_next += (i < 0 || i >= dim || j - 1 < 0 || j - 1 >= dim) ? Vector2f::Zero() : field[IND(i, j - 1, dim)];
    x_next += (i < 0 || i >= dim || j + 1 < 0 || j + 1 >= dim) ? Vector2f::Zero() : field[IND(i, j + 1, dim)];
    x_next += (i < 0 || i >= dim || j < 0 || j >= dim) ? Vector2f::Zero() : Vector2f(field[IND(i, j, dim)](0) * alpha, field[IND(i, j, dim)](1) * alpha);
    x_next /= (4 + alpha);
    field[IND(i - 1, j, dim)] = x_next;
}

/**
 * Perform Jacobi iteration for the diffusion vector
 * @param x is the coordinate/position vector following notation of paper.
 * @param field The relevant scalar field
 * @param div Timestep
 * @param rdx Reciprocal of the grid scale
 * @param dim The maximum dimension of the field [for bound checking]
 * @authors Samaria Mulligan, Patrick Yevych
 */
__device__ void next_poisson(Vector2f x, float *field, float div, float rdx, unsigned dim) {
    int i = (int)x(0);
    int j = (int)x(1);
    float x_next = 0;
    float alpha = -1 * rdx * rdx;
    x_next += (i - 1 < 0 || i - 1 >= dim || j < 0 || j >= dim) ? 0 : field[IND(i - 1, j, dim)];
    x_next += (i + 1 < 0 || i + 1 >= dim || j < 0 || j >= dim) ? 0 : field[IND(i + 1, j, dim)];
    x_next += (i < 0 || i >= dim || j - 1 < 0 || j - 1 >= dim) ? 0 : field[IND(i, j - 1, dim)];
    x_next += (i < 0 || i >= dim || j + 1 < 0 || j + 1 >= dim) ? 0 : field[IND(i, j + 1, dim)];
    x_next += (i < 0 || i >= dim || j < 0 || j >= dim) ? 0 : alpha * div;
    x_next /= 4;
    field[IND(i - 1, j, dim)] = x_next;
}