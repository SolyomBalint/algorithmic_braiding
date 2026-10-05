# Algorithmic Braiding

This project was made for the **Shapr3D Scholarship, spring semester 2026**.

It takes a triangle mesh and computes a field of nearly geodesic curves on its
surface, then extracts them as evenly spaced isolines. These curves are the
first step towards weaving a physical object out of straight ribbons. Results
are shown in an interactive Polyscope viewer.

## The paper

The implementation follows Section 4 of:

> Josh Vekhter, Jiacheng Zhuo, Luisa F. Gil Fandino, Qixing Huang, Etienne Vouga.
> **Weaving Geodesic Foliations.** ACM Transactions on Graphics 38(4), SIGGRAPH 2019.

Only the first part of the paper (the geodesic foliation) is implemented. The
second part, which builds triaxial weaves from a 6-RoSy field and simulates the
ribbons, is not done yet.

## Features

- **Geodesic direction field.** Optimizes a unit vector field per face so that
  its integral curves are close to geodesics (Algorithm 1 of the paper). It
  starts from a smooth field and then gradually enforces the curl constraint.
- **Puncturing.** Detects the singularities of the field and removes the faces
  around them.
- **Integrating factor.** Finds a per face scale `s` that makes the rotated
  field integrable, then rescales it so neighbouring vertices never differ by
  more than half a period.
- **Periodic function θ.** Alternates between solving for θ (as a stripe
  pattern eigenproblem) and refining `s` (Gauss-Newton), so that the level sets
  of θ follow the field.
- **Isoline extraction.** Unwraps θ on each face and traces the leaves, with
  adjustable spacing between them.
- **Interactive GUI.** Every stage can be run on its own or all at once, and
  long solves can be cancelled. The GUI can show the field, its curl,
  singularities and the leaves. Advanced solver parameters can be changed in the
  GUI, and you can switch between the built in presets (torus, aqua-center,
  bunny) or load any `.obj` / `.off` mesh.

## Dependencies

All dependencies are downloaded automatically by
[CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) when you configure the
project.

| Library | Version | Used for | License |
|---|---|---|---|
| [Eigen](https://gitlab.com/libeigen/eigen) | 3.4.0 | Dense and sparse linear algebra | MPL 2.0 |
| [Directional](https://github.com/avaxman/Directional) | 3.0.0 | Mesh data structures, discrete operators, sample meshes | MPL 2.0 |
| [Polyscope](https://github.com/nmwsharp/polyscope) | 2.6.1 | Visualization and GUI (comes with ImGui, glm, glad) | MIT |
| [GLFW](https://github.com/glfw/glfw) | 3.4 | Windowing, built for Wayland only | zlib |

The toolchain (clang 21, CMake 4.1 or newer, Ninja) and the Wayland/OpenGL
runtime libraries come from a [devenv](https://devenv.sh) Nix shell.

## Building and running

```sh
devenv shell
cmake -B build -G Ninja
cmake --build build
./build/braiding
```

Build in Release mode (the default). Without optimizations the sparse solvers
are 20 to 100 times slower.

The default GLFW build has no X11 support, so the viewer needs a Wayland
session.

## Examples

<img width="652" height="971" alt="vase" src="https://github.com/user-attachments/assets/b8f7e030-df97-4914-85b5-f010c3fe8fd8" />

<img width="692" height="920" alt="hand" src="https://github.com/user-attachments/assets/371a7a43-8ad6-425f-9ab4-64584e812a27" />

<img width="771" height="931" alt="pega2" src="https://github.com/user-attachments/assets/8f14e195-12c5-4de6-a20a-a0f1fb917ff3" />

## License

MIT, see [LICENSE](LICENSE).
