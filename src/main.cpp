#include "face_field_operators.h"
#include "geodesic_field.h"
#include "geodesic_optimizer.h"
#include "weaving_mesh.h"

#include <directional/directional_viewer.h>

#include <exception>
#include <filesystem>
#include <memory>
#include <print>
#include <ranges>
#include <string>
#include <vector>

struct Session {
    WeavingMesh weaving_mesh;
    GeodesicField geodesic_field;
    FaceFieldOperators field_operators;
    GeodesicOptimizer geodesic_optimizer;

    explicit Session(std::string const& path)
        : weaving_mesh(path)
        , geodesic_field(weaving_mesh)
        , field_operators(assemble_face_field_operators(weaving_mesh))
        , geodesic_optimizer(geodesic_field, field_operators)
    {
    }
};

std::vector<std::string> list_obj_files()
{
    std::vector<std::string> names;
    std::filesystem::path const data_path(DIRECTIONAL_DATA_PATH);
    if (!std::filesystem::is_directory(data_path)) {
        return names;
    }
    for (std::filesystem::directory_entry const& entry :
        std::filesystem::directory_iterator(data_path)) {
        if (entry.path().extension() == ".obj") {
            names.push_back(entry.path().filename().string());
        }
    }
    std::ranges::sort(names);
    return names;
}

int index_of(std::vector<std::string> const& names, std::string const& name)
{
    auto const it = std::ranges::find(names, name);
    if (it == names.end()) {
        return 0;
    }
    return static_cast<int>(it - names.begin());
}

std::unique_ptr<Session> session = std::make_unique<Session>(
    std::string(DIRECTIONAL_DATA_PATH) + "/fertility.obj");
std::vector<std::string> const obj_files = list_obj_files();
int selected_obj = index_of(obj_files, "fertility.obj");
directional::DirectionalViewer viewer;
polyscope::SurfaceFaceScalarQuantity* curl_quantity = nullptr;
bool singularities_enabled = true;
bool curl_enabled = false;
double smoothness = 100.0;
double lambda_end = 1e-8;
double shrink = 10.0;
double epsilon = 1e-6;
int max_iterations = 50;

void reset_viewer()
{
    polyscope::removeStructure("Mesh 0", false);
    polyscope::removeStructure("Field 0", false);
    polyscope::removeStructure("Singularities 0", false);
    curl_quantity = nullptr;
    viewer.set_surface_mesh(session->weaving_mesh.mesh());
    viewer.set_cartesian_field(session->geodesic_field.field());
    viewer.toggle_singularities(singularities_enabled);
    curl_quantity = viewer.set_surface_face_data(
        session->geodesic_field.face_curl(), "curl");
    curl_quantity->setEnabled(curl_enabled);
}

void refresh_viewer()
{
    session->geodesic_field.update_singularities();
    viewer.set_cartesian_field(session->geodesic_field.field());
    viewer.toggle_singularities(singularities_enabled);
    Eigen::VectorXd const curl = session->geodesic_field.face_curl();
    curl_quantity->updateData(curl);
    curl_quantity->setMapRange({ curl.minCoeff(), curl.maxCoeff() });
}

void load_obj(std::string const& filename)
{
    std::string const path =
        std::string(DIRECTIONAL_DATA_PATH) + "/" + filename;
    try {
        session = std::make_unique<Session>(path);
    } catch (std::exception const& error) {
        std::println("failed to load {}: {}", path, error.what());
        return;
    }
    reset_viewer();
}

void callback()
{
    if (!obj_files.empty()
        && ImGui::BeginCombo("obj", obj_files[selected_obj].c_str())) {
        for (int const i :
            std::views::iota(0, static_cast<int>(obj_files.size()))) {
            bool const selected = i == selected_obj;
            if (ImGui::Selectable(obj_files[i].c_str(), selected)
                && i != selected_obj) {
                selected_obj = i;
                load_obj(obj_files[i]);
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::InputDouble("lambda", &smoothness);
    ImGui::InputDouble("lambda end", &lambda_end);
    ImGui::InputDouble("shrink", &shrink);
    ImGui::InputDouble("epsilon", &epsilon);
    ImGui::InputInt("max iterations", &max_iterations);
    if (ImGui::Button("algorithm 1 step")) {
        (void)session->geodesic_optimizer.step(smoothness);
        refresh_viewer();
    }
    if (ImGui::Button("optimize")) {
        session->geodesic_optimizer.optimize(
            smoothness, lambda_end, shrink, epsilon, max_iterations);
        refresh_viewer();
    }
    if (ImGui::Button("perturb field")) {
        session->geodesic_field.perturb_random();
        refresh_viewer();
    }
    if (ImGui::Button("toggle curl map")) {
        curl_enabled = !curl_quantity->isEnabled();
        curl_quantity->setEnabled(curl_enabled);
    }
    if (ImGui::Button("toggle singularities")) {
        singularities_enabled = !singularities_enabled;
        viewer.toggle_singularities(singularities_enabled);
    }
}

int main()
{
    viewer.init();
    reset_viewer();
    viewer.set_callback(callback);
    viewer.launch();

    return 0;
}
