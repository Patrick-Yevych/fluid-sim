# Fluid Simulation with Navier-Stokes
demo1: https://youtu.be/ZlRuxI-4WbQ

demo2: https://youtube.com/shorts/-zmT2wl-XpA

## Inspiration

The inspiration behind fluid-sim was to create a tool that could help us better understand and simulate the behavior of air, smoke, and smog - complex fluids that play a critical role in our environment and our health. By simulating the movement and interactions of these fluids, we hope to gain insight on the effects of climate change and develop more effective strategies for modeling and predicting it.

## How to run
This is a CUDA, Eigen, SFML, and OpenGL application. To compile, simply run 

```
$ nvcc fluids.cu -o ./out -lglfw -lGLU -lGL
```

The executable file (call it `out`), just run `./out`. A GUI displaying a 512 × 512 field will be displayed, where you can interactively apply a force to see the action. Areas of high velocity are labelled yellow, followed by green, blue, and then purple.

You can also adjust the timestep, fluid viscosity, applied force decay, and the radius of the applied force for a more customized experience.

```
$ ./out TIMESTEP VISCOSITY DECAY RADIUS
```

The terminal will periodically print out the velocity of the central point of the field, and the current strength of the applied force.

## What it does

Our program simulates the effect of temperature change on the state of fluids of various viscosity such as water, air, smoke, and smog. Users can configure simulation parameters such as fluid viscosity, simulation granularity, force multiplier, radius, and decay rate. Additionally, users can generate temperature change by clicking and dragging the mouse across the window and see a real time visualization of the fluid.

| Timesteps | Viscosity | Decay | Description              | Pictures             |
| --------- | --------- | ----- | ------------------------ | -------------------- |
| 0.25      | 1         | 1     | Air-like fluid (Default) | ![](./Demos/Default) |
| 0.1       | 10        | 5     | Smog-like fluid          | ![](./Demos/muddy)   |
|           |           |       |                          |                      |

## How we built it

To build our fluid simulation, we leveraged the libraries CUDA, OpenGL, and Eigen in C++. We began by studying Stable Fluids, a [paper](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf) published by Joe Stam of the University of Toronto. This paper presents a method to simulate fluids by discretizing and numerically computing the Navier Stokes equations. So, we encapsulate the state of the fluid by a velocity vector field and a scalar pressure field. These are represented as arrays of Eigen vectors. Next, we invoke a CUDA kernal to compute the velocity and pressure values at each pixel through a five step process: self-advection, diffusion, force application, Poisson-pressure, and projection. Afterwards, we use another CUDA kernal to convert the velocity vectors into RGB colour values and render the visualization with OpenGL.

## Challenges we ran into

The first major challenged we encountered was interpreting the mathematical equations presented by the paper and converting into a numerical algorithm. This took the collective efforts and the combined expertise of each team member. This served as the first major obstacle which took most of the first night to complete.

The second major challenge was mapping the velocity vector field to a set of RGB values, rendering and animating it on to screen. This involved a lengthy process of configuring CUDA-OpenGL interop and researching numerous documentations. More specifically, the largest hurdle was insuring the RGB texture can be read and displayed by the OpenGL library used.

Finally, the most stubborn and time consuming issue was solving the NaN (not a number) bug that originated from the force application function and propagated across the velocity vector field. This involved us debugging several of the helper functions, as well as each step of the process in order to locate and solve the issue. Afterwards, the final effort was tweaking the simulation parameters to a more realistic configuration.

## Accomplishments that we're proud of

The accomplishment that we're most proud of was our ability to interpret the difficult mathematics behind the Navier Stokes equations. This section by far demanded the most collaboration amongst the team members as it required each member to share their interpretation and intuition with others.

We're also proud of the achievement that we were able to delve into a unique different topic that has real world applications in physics simulations and that was challenging. This program also involved a lot of creativity and problem solving skills, as we had to work with many aspects of computer science, including low level code, math, physics, multiprocessing, and compute graphics. Learning and overcoming these aspects became one of our proudest achievements.

## What we learned

This project improved our technical knowledge of computer graphics as well as our theoretical knowledge of fluid mechanics. More specifically, we gained more experience in C++ development with industry standard libraries for GPU accelerated workloads and massive matrix operations. These tools and technologies are commonly used in fields of computer graphics such as animation and game development, as well as fields using high performance computing such as financial technology and machine learning.

Furthermore, this project also helps us to reinforce our skills in linear algebra and multivariable calculus, which will be useful in further academic endeavors.

## What's next for Fluid Sim

During our research, we encountered other papers such as [Real-time Simulation of Large Bodies of Water
with Small Scale Details](https://diglib.eg.org/xmlui/bitstream/handle/10.2312/SCA.SCA10.197-206/197-206.pdf?sequence=1&isAllowed=y), which presented potential improvements to the realism of fluid simulations. More specifically the paper notes ways to reduce overshooting and artificing, as well as methods of simulating large bodies of 3D fluids.

Furthermore, we also noted that web assembly includes GPU rendering workloads in certain experimental builds, so providing a portable web demo to the end users will improve user experience.
