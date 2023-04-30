# Fluid Simulation

## Code

### Packages
External packages required besides Cuda and its prerequisites are

1. OpenGL
2. SFML
3. Eign3

### Constants
The variables we will fix will be

* Time Step
* Dimension of Square
* Resolution of Square
* Viscosity
* Radius
* Decay Rate

We will also define two auxiliary functions to accommodate for common coding constructs and edge cases

* `IND(x, y, d)` converts 2D coordinates into 1D coordinates, provided you have the length of the array.
* `CLAMP(x)` takes all out-of-bounds values to their corresponding at-the-bounds values.
    * `x < 0.` will result in `x` being set to nil.
    * `x > 1.` will result in `x` being set to unity.

Two global constants, both of type `Eigen::Vector2f` will be involved, namely:

* `C`, which varies by mouse click location
* `F`, which varies by mouse drag. It is a vector value.

### Field Initialization
The `initializeField` of type `T` initializes a pointer of a vector field (as a 1D array) to the device. The initialization will be a preset function.

The `mouse_button_callback` function is a callback which response to mouse clocks and drags. These functions affect the global `C` and `F` values.

### Decay Force
The decay rate is a flat 0.01 reduction of the values across both components.

### Bilerp function
Helper function to perform bilinear interpolation on indices between integers.