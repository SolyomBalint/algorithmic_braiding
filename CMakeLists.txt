cmake_minimum_required(VERSION 4.1)

project(algorithmic_braiding LANGUAGES C CXX)

# Eigen is header-only template code: an unoptimized build (no -O, no NDEBUG)
# makes its sparse factorizations 20-100x slower. Default to Release unless the
# user picked something else (single-config generators only).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Several bundled dependencies (GLFW 3.4, glm, etc.) still declare a
# cmake_minimum_required below 3.5, which CMake 4.x rejects outright. Treat any
# such request as 3.5 so these third-party trees keep configuring.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

option(
    USE_CPM_FOR_DEPENDENCIES
    "Use the CPM package manager to automatically download the project's dependencies"
    ON
)

message(STATUS "Setting CMAKE_DEVENV_PROFILE to $ENV{DEVENV_PROFILE}")
set(CMAKE_DEVENV_PROFILE $ENV{DEVENV_PROFILE})

if(USE_CPM_FOR_DEPENDENCIES)
    set(CPM_VERSION "v0.42.3")
    set(CPM_HASH "a609e875fd532b067174250f6abbc3dac22fe2d64869783fb1e80bda1625c844")
    set(CPM_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake")

    if(NOT EXISTS ${CPM_PATH})
        message(STATUS "Downloading CPM version ${CPM_VERSION}")
        file(
            DOWNLOAD
            https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_VERSION}/CPM.cmake
            ${CPM_PATH}
            EXPECTED_HASH SHA256=${CPM_HASH}
        )
    else()
        message(STATUS "CPM already downloaded")
    endif()

    include(${CPM_PATH})

    # --- Eigen ----------------------------------------------------------------
    # Header-only: we just need its sources on an include path. Both our own code
    # and Directional link against this exact configuration.
    CPMAddPackage(
        NAME eigen
        GITLAB_REPOSITORY libeigen/eigen
        GIT_TAG 3.4.0
        DOWNLOAD_ONLY YES
    )

    # --- GLFW (override Polyscope's bundled 3.3.9) ----------------------------
    # Polyscope ships GLFW 3.3.9 as a submodule whose Wayland support is
    # experimental (EGL-only, undecorated windows, needs extra-cmake-modules).
    # We pull GLFW 3.4 first so Polyscope's `if(NOT TARGET glfw)` guard picks up
    # this newer tree, which has mature native-Wayland + libdecor decorations.
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND ON CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)
    CPMAddPackage(
        NAME glfw
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.4
    )

    # --- Polyscope ------------------------------------------------------------
    # Brings the rest of its bundled deps (glm, imgui, glad, ...) via submodules,
    # which CPM fetches automatically. Reuses the `glfw` target defined above.
    CPMAddPackage(
        NAME polyscope
        GITHUB_REPOSITORY nmwsharp/polyscope
        GIT_TAG v2.6.1
    )

    # Polyscope's bundled imgui (1.92.x) GLFW backend defaults to the X11 native
    # API on Linux, which drags in <X11/Xlib.h>. Since we build GLFW Wayland-only,
    # tell imgui to skip its X11 path so no X11 dev headers are needed.
    if(TARGET imgui)
        target_compile_definitions(imgui PUBLIC IMGUI_IMPL_GLFW_DISABLE_X11)
    endif()

    # --- Directional ----------------------------------------------------------
    # Header-only library. We DOWNLOAD_ONLY and never add_subdirectory it, so its
    # vendored eigen and its polyscope submodule are fetched but never built or
    # linked (CPM doesn't honor GIT_SUBMODULES here, and eigen is committed into
    # the repo regardless). We consume only its include/ directory below, against
    # the Eigen and Polyscope this project configures itself.
    CPMAddPackage(
        NAME Directional
        GITHUB_REPOSITORY avaxman/Directional
        GIT_TAG v3.0.0
        DOWNLOAD_ONLY YES
    )
endif()

# Eigen interface target consumed by both our code and Directional.
add_library(Eigen3_Eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
target_include_directories(Eigen3_Eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")

# Directional interface target (headers + our Eigen).
add_library(directional INTERFACE)
target_include_directories(directional SYSTEM INTERFACE "${Directional_SOURCE_DIR}/include")
target_link_libraries(directional INTERFACE Eigen3::Eigen)

add_executable(braiding src/main.cpp)
target_link_libraries(braiding PRIVATE Eigen3::Eigen polyscope directional)

# Path to Directional's bundled tutorial meshes so the demo can load one.
target_compile_definitions(
    braiding
    PRIVATE DIRECTIONAL_DATA_PATH="${Directional_SOURCE_DIR}/tutorial/data"
)
