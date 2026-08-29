#include "face_field_operators.h"
#include "geodesic_field.h"
#include "weaving_mesh.h"

#include <directional/directional_viewer.h>

#include <string>

WeavingMesh weaving_mesh(std::string(DIRECTIONAL_DATA_PATH) + "/fertility.obj");
GeodesicField geodesic_field(weaving_mesh);
FaceFieldOperators field_operators
    = assemble_face_field_operators(weaving_mesh);
directional::DirectionalViewer viewer;
polyscope::SurfaceFaceScalarQuantity* curl_quantity = nullptr;

void callback()
{
    if (ImGui::Button("perturb field")) {
        geodesic_field.perturb_random();
        viewer.set_cartesian_field(geodesic_field.field());
        Eigen::VectorXd const curl = geodesic_field.face_curl();
        curl_quantity->updateData(curl);
        curl_quantity->setMapRange({ curl.minCoeff(), curl.maxCoeff() });
    }
    if (ImGui::Button("toggle curl map")) {
        curl_quantity->setEnabled(!curl_quantity->isEnabled());
    }
}

int main()
{
    (void)field_operators;

    viewer.init();
    viewer.set_surface_mesh(weaving_mesh.mesh());
    viewer.set_cartesian_field(geodesic_field.field());
    curl_quantity = viewer.set_surface_face_data(
        geodesic_field.face_curl(), "curl");
    curl_quantity->setEnabled(false);
    viewer.set_callback(callback);
    viewer.launch();

    return 0;
}
