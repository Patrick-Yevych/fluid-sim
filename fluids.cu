#include <iostream>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <GLFW/glfw3.h>
#include "tinycolormap.hpp"

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <unistd.h> // for sleep function. use window.h for windows.
#endif

#define TIMESTEP 0.25
#define DIM 1024
#define RES 1024
#define VISCOSITY 1
#define RADIUS 1
#define DECAY_RATE 0.01

#define IND(x, y, d) int((y) * (d) + (x))

using namespace std;
using Eigen::Vector2f;
using Eigen::Vector3f;

// mouse click location
Vector2f C;
// direction and length of mouse drag
Vector2f F;

template <typename T>
void initializeField(T **f, T **dev_f, T val, unsigned dim) {
    *f = (T *)malloc(dim * dim * sizeof(T));
    cudaMalloc(dev_f, dim * dim * sizeof(T));
    for (int i = 0; i < dim*dim; i++) *(*f + i) = val;
    cudaMemcpy(*dev_f, *f, dim * dim * sizeof(T), cudaMemcpyHostToDevice);
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    double xpos, ypos, xend, yend, xdir, ydir, len;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        glfwGetCursorPos(window, &xpos, &ypos);
        C << (int)xpos, (int)ypos;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        glfwGetCursorPos(window, &xend, &yend);
        xdir = xend - xpos;
        ydir = yend - ypos;
        F = Vector2f(xdir, ydir).normalized();
    }
}


void decayForce() {
    float nx = F(0) - DECAY_RATE;
    float ny = F(1) - DECAY_RATE;
    nx = (nx > 0) ? nx : 0;
    ny = (ny > 0) ? ny : 0;
    F << nx, ny;
}


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
        return 0;

    Vector2f wL = (i - 1 < 0) ? Vector2f::Zero() : from[IND(i - 1, j, dim)];
    Vector2f wR = (i + 1 >= dim) ? Vector2f::Zero() : from[IND(i + 1, j, dim)];
    Vector2f wB = (j - 1 < 0) ? Vector2f::Zero() : from[IND(i, j - 1, dim)];
    Vector2f wT = (j + 1 <= dim) ? Vector2f::Zero() : from[IND(i, j + 1, dim)];

    return halfrdx * (wR(0) - wL(0), wT(1) - wB(1));
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
    field[IND(i, j, dim)] += F * pow(timestep, exp);
}

/***
 * Navier-Stokes computation kernel.
*/
__global__ void nskernel(Vector2f* u, float* p, float rdx, float viscosity, Vector2f c, Vector2f F, int timestep, float r, unsigned dim)
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


__device__ Vector3f getColor(double val, tinycolormap::ColormapType type) {
    tinycolormap::Color color = tinycolormap::GetColor(val, type);
    Vector3f ret(color.r(), color.g(), color.b());
    return ret;
}

/***
 * color mapping kernel.
*/
__global__ void clrkernel(Vector3f *uc, Vector2f *u, unsigned dim) {
    Vector2f x(threadIdx.x, threadIdx.y);
    uc[IND(x(0), x(1), dim)] = getColor(
                                    (double)u[IND(x(0), x(1), dim)].norm(), 
                                    tinycolormap::ColormapType::Viridis);
}

// render points using VBO
void renderPointsVBO(GLuint vbo, int num_points) {
    // Bind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Set up vertex attribute pointer
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    glEnableClientState(GL_VERTEX_ARRAY);

    // Set point size and color
    glPointSize(3.0f);
    glColor3f(1.0f, 0.0f, 0.0f);

    // Draw the points using the VBO
    glDrawArrays(GL_POINTS, 0, num_points);

    // Disable the vertex attribute pointer and unbind the VBO
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int main(void) {

    // quarter of second timestep
    float timestep = TIMESTEP;
    // dimension of vector fields
    unsigned dim = DIM;
    // resolution of display
    unsigned res = RES;
    // how many pixels a cell of the vector field represents
    float rdx = res / dim;

    // fluid parameters
    float viscosity = VISCOSITY;

    // force parameters
    C = Vector2f::Zero();
    F = Vector2f::Zero();
    float r = RADIUS;

    // fluid state representation: 
    // velocity vector field (u) and pressure scalar field (p).
    Vector2f *u, *dev_u;
    float *p, *dev_p;

    initializeField<Vector2f>(&u, &dev_u, Vector2f::Zero(), dim);
    initializeField<float>(&p, &dev_p, 0, dim);

    // color maps
    Vector3f *uc, *dev_uc;
    initializeField<Vector3f>(&uc, &dev_uc, Vector3f::Zero(), dim);


    // initialize glfw
    if (!glfwInit())
        return -1;

    // create glfw window
    GLFWwindow* window = glfwCreateWindow(res, res, "Fluid Sim", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Create a VBO for the points (Vector3f*)
    GLuint vbo;
    // Generate a VBO handle
    glGenBuffers(1, vbo);
    // Bind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    // Copy data to the VBO
    glBufferData(GL_ARRAY_BUFFER, num_of_points * sizeof(Vector3), *(uc[0]), GL_STATIC_DRAW);
    // Unbind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // main loop
    dim3 threads(dim, dim);
    while (!(glfwWindowShouldClose(window))) {
        nskernel<<<1, threads>>>(dev_velocity, dev_pressure, rdx, viscosity, c, F, timestep, r, dim);
        cudaDeviceSynchronize();
        clrkernel<<<1, threads>>>(dev_uc, dev_u, dim);
        cudaDeviceSynchronize();

        cudaMemcpy(*uc, *dev_uc, dim * dim * sizeof(Vector3f), cudaMemcpyDeviceToHost);

        decayForce();

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // Render the points using VBO
        renderPointsVBO(vbo, points.size());

        // update screen
        glfwSwapBuffers(window);
        // for the mouse event
        glfwPollEvents();
    }

    // Cleanup
    glDeleteBuffers(1, &vbo);
    glfwTerminate();
}
