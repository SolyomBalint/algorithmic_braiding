// Basic Directional visualization demo (mirrors Directional tutorial 101,
// "Glyph Rendering"): load a triangle mesh, attach a precomputed face-based
// tangent vector field plus its singularities, and render it interactively
// through Directional's Polyscope-backed viewer.
//
// Everything here builds against the Eigen and Polyscope that *this* project
// pulls in via CPM; Directional is consumed header-only on top of them.

#include <directional/CartesianField.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/TriMesh.h>
#include <directional/directional_viewer.h>
#include <directional/readOFF.h>
#include <directional/read_raw_field.h>
#include <directional/read_singularities.h>

int main()
{
    directional::TriMesh mesh;
    directional::PCFaceTangentBundle ftb;
    directional::CartesianField field;
    directional::DirectionalViewer viewer;

    int N = 0; // degree of the field (filled in by read_raw_field)

    // Load the mesh and a precomputed N-RoSy field defined on its faces.
    directional::readOFF(DIRECTIONAL_DATA_PATH "/bumpy.off", mesh);
    ftb.init(mesh);
    directional::read_raw_field(DIRECTIONAL_DATA_PATH "/bumpy.rawfield", ftb, N, field);
    directional::read_singularities(DIRECTIONAL_DATA_PATH "/bumpy.sings", field);

    // Hand it all to Directional's viewer, which drives Polyscope under the hood.
    viewer.init();
    viewer.set_surface_mesh(mesh);
    viewer.set_cartesian_field(field);
    viewer.launch(); // opens the window; blocks until closed

    return 0;
}
