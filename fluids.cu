// A DeerHacks last-hour khuya submission.
// nvcc fluids.cu -o ./out -lglfw -lGLU -lGL
#include "const.h"

// mouse click location
Vector2f C;
// direction and length of mouse drag
Vector2f F;
// decay rate
float global_decay_rate = DECAY_RATE;

/**
 * Initializes a vector or scalar field with initial conditions to both
 * the hostside and deviceside.
 * @param f The field on the host.
 * @param dev_f The field on the device.
 * @param val Initial conditions.
 * @param dim The dimensions [for boundary checking]
 * @authors Patrick Yevych
 */
template <typename T>
void initializeField(T **f, T **dev_f, T val, unsigned dim)
{
    *f = (T *)malloc(dim * dim * sizeof(T));
    cudaMalloc(dev_f, dim * dim * sizeof(T));
    for (int i = 0; i < dim * dim; i++)
        *(*f + i) = val;
    cudaMemcpy(*dev_f, *f, dim * dim * sizeof(T), cudaMemcpyHostToDevice);
}

/**
 * Called whenever a click or release even happens on the window.
 * Updates the convection F and the mouse click location C.
 * @param window The GLFWwindow object to be applied to
 * @param button ID of the clickable clicked
 * @param action The type of action registered
 * @param mods Any specific mods applied to this action
 * @authors Patrick Yevych
 */
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    double xpos, ypos, xend, yend;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        glfwGetCursorPos(window, &xpos, &ypos);
        C = Vector2f((int)xpos, (int)ypos);
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        glfwGetCursorPos(window, &xend, &yend);
        F = Vector2f(xend - C(0), yend - C(1));
    }
}

/**
 * Decays the convection force F.
 * @authors Patrick Yevych
 */
void decayForce()
{
    float nx = F(0) - global_decay_rate;
    float ny = F(1) - global_decay_rate;
    nx = (nx > 0) ? nx : 0;
    ny = (ny > 0) ? ny : 0;
    F = Vector2f(nx, ny);
}

/**
 * Implementation of bilinear interpolation given input location.
 * @param pos The input location, supporting intermediate positions
 * @param field The vector field
 * @param dim The dimensions [for boundary checking]
 * @authors Alex Apostolou, Samaria Mulligan
 * @link https://en.wikipedia.org/wiki/Bilinear_interpolation
 */
__device__ Vector2f bilerp(Vector2f pos, Vector2f *field, unsigned dim)
{
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

/**
 * Obtain the approximate divergence of a vector field.
 * The divergence is calculated using the immediate neighboring value only
 * across the four cardinal directions.
 * @param x Cartesian location of the field
 * @param from The vector field
 * @param halfrdx Half of the rdx value [for efficiency reasons]
 * @param dim The maximum dimension of the field [for bound checking]
 * @return The approximate divergence value
 * @authors Alex Apostolou
 */
__device__ float divergence(Vector2f x, Vector2f *from, float halfrdx, unsigned dim)
{
    int i = x(0);
    int j = x(1);
    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return 0;

    Vector2f wL = (i - 1 < 0) ? Vector2f::Zero() : from[IND(i - 1, j, dim)];
    Vector2f wR = (i + 1 >= dim) ? Vector2f::Zero() : from[IND(i + 1, j, dim)];
    Vector2f wB = (j - 1 < 0) ? Vector2f::Zero() : from[IND(i, j - 1, dim)];
    Vector2f wT = (j + 1 >= dim) ? Vector2f::Zero() : from[IND(i, j + 1, dim)];

    return halfrdx * (wR(0) - wL(0), wT(1) - wB(1));
}

/**
 * Obtain the approximate gradient of a scalar field [in this case, p].
 * The gradient is calculated using the immediate neighboring value only.
 * @param x Cartesian location of the field
 * @param p The scalar field [pressure]
 * @param halfrdx Half of the rdx value [for efficiency reasons]
 * @param dim The maximum dimension of the field [for bound checking]
 * @return The approximate gradient, as a Vector2f
 * @authors Alex Apostolou
 */
__device__ Vector2f gradient(Vector2f x, float *p, float halfrdx, unsigned dim)
{
    int i = x(0);
    int j = x(1);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return Vector2f::Zero();

    float pL = (i - 1 < 0) ? 0 : p[IND(i - 1, j, dim)];
    float pR = (i + 1 >= dim) ? 0 : p[IND(i + 1, j, dim)];
    float pB = (j - 1 < 0) ? 0 : p[IND(i, j - 1, dim)];
    float pT = (j + 1 >= dim) ? 0 : p[IND(i, j + 1, dim)];

    return halfrdx * Vector2f(pR - pL, pT - pB);
}

/***
 * Computes the advection of the fluid.
 * @param x is the coordinate/position vector following notation of chp 38.
 * @param velfield is u, the velocity field as of the current time quanta.
 * @param field is the current field being updated.
 * @param timestep delta t for next iteration
 * @param rdx approximation constant
 * @param dim The maximum dimension of the field [for bound checking]
 * @authors Patrick Yevych
 */
__device__ void advect(Vector2f x, Vector2f *field, Vector2f *velfield, float timestep, float rdx, unsigned dim)
{
    Vector2f pos = x - timestep * rdx * velfield[IND(x(0), x(1), dim)];
    field[IND(x(0), x(1), dim)] = bilerp(pos, field, dim);
}

/**
 * Generalized Jacobi for computing pressure or viscous diffusion of fluid.
 * @param x is the coordinate/position vector following notation of paper.
 * @param field The relevant vector field
 * @param alpha rdx*rdx/(viscosity*timestep) for diffusion; -1*timestep*timestep for pressure.
 * @param beta 4+alpha for diffusion; 4 for pressure.
 * @param b u(x) for diffusion; divergence for pressure.
 * @param dim The maximum dimension of the field [for bound checking].
 * @authors Patrick Yevych
 */
template <typename T>
__device__ void jacobi(Vector2f x, T *field, float alpha, float beta, T b, T zero, unsigned dim)
{
    int i = (int)x(0);
    int j = (int)x(1);

    T f00 = (i - 1 < 0 || i - 1 >= dim || j < 0 || j >= dim) ? zero : field[IND(i - 1, j, dim)];
    T f01 = (i + 1 < 0 || i + 1 >= dim || j < 0 || j >= dim) ? zero : field[IND(i + 1, j, dim)];
    T f10 = (i < 0 || i >= dim || j - 1 < 0 || j - 1 >= dim) ? zero : field[IND(i, j - 1, dim)];
    T f11 = (i < 0 || i >= dim || j + 1 < 0 || j + 1 >= dim) ? zero : field[IND(i, j + 1, dim)];
    T ab = (i < 0 || i >= dim || j < 0 || j >= dim) ? zero : alpha * b;

    field[IND(i-1, j, dim)] = (f00 + f01 + f10 + f11 + ab) / beta;
}

/**
 * Apply the external source to the deviceside data
 * @param x is the coordinate/position vector following notation of chp 38.
 * @param field The relevant vector field
 * @param C The center of the applied force
 * @param F The value of the applied force
 * @param timestep The time step per iteration of the program
 * @param r The radius of the applied force
 * @param dim The maximum dimension of the field [for bound checking]
 * @authors Patrick Yevych, Hong Wei, Samaria Mulligan
 */
__device__ void force(Vector2f x, Vector2f *field, Vector2f C, Vector2f F, float timestep, float r, unsigned dim)
{
    float xC[2] = {x(0) - C(0), x(1) - C(1)};
    float exp = (xC[0] * xC[0] + xC[1] * xC[1]) / r;
    int i = x(0);
    int j = x(1);
    Vector2f temp = F * timestep * pow(2.718, exp) * 0.001;
    field[IND(i, j, dim)] += F * timestep * pow(2.718, exp) * 0.001;
    if ((temp(0) != 0 || temp(1) != 0) && x(0) == DIM / 2 && x(1) == DIM / 2)
        printf("G1 = (%f, %f)\n", temp(0), temp(1));
}

/**
 * Navier-Stokes computation kernel.
 * @param u The vector velocity field
 * @param p The scalar pressure field
 * @param rdx Reciprocal of the grid scale
 * @param viscosity The viscosity of the fluid
 * @param C The center of the applied force
 * @param F The value of the applied force
 * @param timestep The time step per iteration of the program
 * @param r The radius of the applied force
 * @param dim The maximum dimension of the field [for bound checking]
 * @authors Patrick Yevych
 */
__global__ void nskernel(Vector2f *u, float *p, float rdx, float viscosity, Vector2f C, Vector2f F, float timestep, float r, unsigned dim)
{
    Vector2f x(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);

    // advection
    advect(x, u, u, timestep, rdx, dim);
    __syncthreads();

    // diffusion
    float alpha = rdx * rdx / (viscosity * timestep), beta = 4 + alpha;
    jacobi<Vector2f>(x, u, alpha, beta, u[IND(x(0), x(1), dim)], Vector2f::Zero(), dim);
    __syncthreads();

    // force application
    force(x, u, C, F, timestep, r, dim);
    __syncthreads();

    // pressure
    alpha = -1 * rdx * rdx; beta = 4;
    jacobi<float>(x, p, alpha, beta, divergence(x, u, (float)(rdx / 2), dim), 0, dim);
    __syncthreads();

    // u = w - nabla p
    u[IND(x(0), x(1), dim)] -= gradient(x, p, (float)(rdx / 2), dim);
    __syncthreads();

    // print state
    if (x(0) == DIM / 2 && x(1) == DIM / 2)
        printf("u[%.1f, %.1f] = (%f, %f)\n", x(0), x(1), u[IND(x(0), x(1), dim)](0), u[IND(x(0), x(1), dim)](1));
}

/**
 * Given the value of x, obtain corresponding RGB value, for visualization.
 * Adapted from Yuki Koyama.
 * @param x The corresponding intermediate value
 * @authors Hong Wei, Alex Apostolou
 * @link https://github.com/yuki-koyama/tinycolormap
 */
__device__ Vector3f getColor(double x)
{
    double data[][3] = VIRIDIS;

    const double a = CLAMP(x) * 255;
    const double i = std::floor(a);
    const double t = a - i;
    auto d0 = data[static_cast<std::size_t>(std::ceil(a))];
    Vector3f c0(d0[0], d0[1], d0[2]);
    auto d1 = data[static_cast<std::size_t>(std::ceil(a))];
    Vector3f c1(d1[0], d1[1], d1[2]);

    return (1.0 - t) * c0 + t * c1;
}

/**
 * Maps velocity vectors to a color
 * @param uc Array of RGB values for every pixel
 * @param u The velocity vector at that location
 * @param dim The maximum dimension of the field [for bound checking]
 * @authors Patrick Yevych
 */
__global__ void clrkernel(Vector3f *uc, Vector2f *u, unsigned dim)
{
    Vector2f x(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    uc[IND(x(0), x(1), dim)] = getColor(
        (double)u[IND(x(0), x(1), dim)].norm());
}

/**
 * Driver code containing the CUDA kernels and OpenGL rendering.
 * @authors Patrick Yevych, Hong Wei
 */
int main(int argc, char **argv)
{
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
    // force decay rate
    global_decay_rate = DECAY_RATE;
    // force radius
    float r = RADIUS;

    // user provided simulation parameters
    if (argc == 5) {
        timestep = atof(argv[1]);
        viscosity = atof(argv[2]);
        global_decay_rate = atof(argv[3]);
        r = atof(argv[4]);
    }
    else if (argc != 1) {
        printf("USAGE: ./out TIMESTEP VISCOSITY DECAY RADIUS\n");
        return 1;
    }

    // force parameters
    C = Vector2f::Zero(); F = Vector2f::Zero();

    // fluid state representation:
    // velocity vector field (u) and pressure scalar field (p).
    Vector2f *u, *dev_u;
    float *p, *dev_p;

    initializeField<Vector2f>(&u, &dev_u, Vector2f::Zero(), dim);
    initializeField<float>(&p, &dev_p, 0, dim);

    // color maps
    Vector3f *uc, *dev_uc;
    initializeField<Vector3f>(&uc, &dev_uc, Vector3f::Zero(), dim);

    // Initialize GLFW
    if (!glfwInit())
        return -1;

    // Create a window
    GLFWwindow *window = glfwCreateWindow(dim, dim, "sim", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    // Make the window's context current
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    // Setup the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, dim, 0, dim, -1, 1);

    // Set up the modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Load the texture from data
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, dim, dim, 0, GL_RGB, GL_FLOAT, uc);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Enable texturing
    glEnable(GL_TEXTURE_2D);

    // Set the texture as the current texture
    glBindTexture(GL_TEXTURE_2D, tex);

    // Set the texture environment parameters
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 blocks(dim / BLOCKSIZEX, dim / BLOCKSIZEY);
    // Loop until the user closes
    while (!glfwWindowShouldClose(window)) {

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, dim, dim, 0, GL_RGB, GL_FLOAT, uc);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);

        glClear(GL_COLOR_BUFFER_BIT);
        // Draw a quad with texture coordinates
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2i(0, 0);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2i(dim, 0);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2i(dim, dim);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2i(0, dim);
        glEnd();

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();

        // compute navier-stokes and colorize
        nskernel<<<blocks, threads>>>(dev_u, dev_p, rdx, viscosity, C, F, timestep, r, dim);
        cudaDeviceSynchronize();
        clrkernel<<<blocks, threads>>>(dev_uc, dev_u, dim);
        cudaDeviceSynchronize();
        cudaMemcpy(uc, dev_uc, dim * dim * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        decayForce();
    }

    free(u);
    free(p);
    free(uc);
    cudaFree(dev_u);
    cudaFree(dev_p);
    cudaFree(dev_uc);

    glfwTerminate();
}