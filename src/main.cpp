#include "face_field_operators.h"
#include "geodesic_field.h"
#include "geodesic_optimizer.h"
#include "weaving_mesh.h"

#include <directional/directional_viewer.h>

#include <string>

WeavingMesh weaving_mesh(std::string(DIRECTIONAL_DATA_PATH) + "/torus.obj");
GeodesicField geodesic_field(weaving_mesh);
FaceFieldOperators field_operators
    = assemble_face_field_operators(weaving_mesh);
GeodesicOptimizer geodesic_optimizer(geodesic_field, field_operators);
directional::DirectionalViewer viewer;
polyscope::SurfaceFaceScalarQuantity* curl_quantity = nullptr;
double smoothness = 100.0;
double lambda_end = 1e-8;
double shrink = 10.0;
double epsilon = 1e-6;
int max_iterations = 50;

void refresh_viewer()
{
    viewer.set_cartesian_field(geodesic_field.field());
    Eigen::VectorXd const curl = geodesic_field.face_curl();
    curl_quantity->updateData(curl);
    curl_quantity->setMapRange({ curl.minCoeff(), curl.maxCoeff() });
}

void callback()
{
    ImGui::InputDouble("lambda", &smoothness);
    ImGui::InputDouble("lambda end", &lambda_end);
    ImGui::InputDouble("shrink", &shrink);
    ImGui::InputDouble("epsilon", &epsilon);
    ImGui::InputInt("max iterations", &max_iterations);
    if (ImGui::Button("algorithm 1 step")) {
        (void)geodesic_optimizer.step(smoothness);
        refresh_viewer();
    }
    if (ImGui::Button("optimize")) {
        geodesic_optimizer.optimize(
            smoothness, lambda_end, shrink, epsilon, max_iterations);
        refresh_viewer();
    }
    if (ImGui::Button("perturb field")) {
        geodesic_field.perturb_random();
        refresh_viewer();
    }
    if (ImGui::Button("toggle curl map")) {
        curl_quantity->setEnabled(!curl_quantity->isEnabled());
    }
}

int main()
{
    viewer.init();
    viewer.set_surface_mesh(weaving_mesh.mesh());
    viewer.set_cartesian_field(geodesic_field.field());
    curl_quantity
        = viewer.set_surface_face_data(geodesic_field.face_curl(), "curl");
    curl_quantity->setEnabled(false);
    viewer.set_callback(callback);
    viewer.launch();

    return 0;
}
