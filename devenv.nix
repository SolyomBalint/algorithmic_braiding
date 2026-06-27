{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  cachix.enable = true;
  cachix.pull = [ "pre-commit-hooks" ];

  env.CPM_SOURCE_CACHE = "${config.devenv.root}/.cache";

  packages = with pkgs; [
    gnumake
    ninja
    cmake
    clang-tools
    clang
    pkg-config

    ## Windowing / GL stack for Polyscope's GLFW (native Wayland backend, GLFW 3.4)
    libGL
    wayland
    wayland-protocols
    wayland-scanner
    libxkbcommon
    libdecor

    ## Tools
    doxygen
    gdb
    valgrind
  ];

  enterShell = ''
    export CC=clang
    export CXX=clang++

    # GLFW dlopen()s its Wayland/EGL backend libraries by soname at runtime, and
    # Nix doesn't place those on the loader path. Prepend them (and the system
    # Mesa GL/EGL driver, which must match the running compositor) while keeping
    # whatever devenv already set.
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:${lib.makeLibraryPath (with pkgs; [ libGL wayland libxkbcommon libdecor ])}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    echo ""
    echo "██████╗ ███╗   ███╗███████╗    ██╗██╗████████╗"
    echo "██╔══██╗████╗ ████║██╔════╝    ██║██║╚══██╔══╝"
    echo "██████╔╝██╔████╔██║█████╗      ██║██║   ██║   "
    echo "██╔══██╗██║╚██╔╝██║██╔══╝      ██║██║   ██║   "
    echo "██████╔╝██║ ╚═╝ ██║███████╗    ██║██║   ██║   "
    echo "╚═════╝ ╚═╝     ╚═╝╚══════╝    ╚═╝╚═╝   ╚═╝   "
    echo ""

    # Check git commit template
    echo "Checking whether git commit template is set for local project"
    if git config --get commit.template >/dev/null 2>&1; then
        echo "Commit template is already set, continuing"
    else
        TEMPLATE_CMD="git config commit.template .git_commit_msg_template"
        echo "No commit template configured, setting it for local project"
        echo "Executing: $TEMPLATE_CMD"
        $TEMPLATE_CMD
    fi
    echo
  '';
}
