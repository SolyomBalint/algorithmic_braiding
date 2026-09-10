#include "face_field_operators.h"
#include "foliation.h"
#include "geodesic_field.h"
#include "geodesic_optimizer.h"
#include "isolines.h"
#include "punctured_mesh.h"
#include "scale_field.h"
#include "weaving_mesh.h"

#include <directional/directional_viewer.h>

#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <string>
#include <vector>

struct Session {
    WeavingMesh weaving_mesh;
    GeodesicField geodesic_field;
    FaceFieldOperators field_operators;
    GeodesicOptimizer geodesic_optimizer;

    // Stages 2–5 of §4.2.5, filled in by the foliation buttons.
    std::optional<PuncturedMesh> punctured;
    FaceFieldOperators punctured_operators;
    Eigen::VectorXd scale; // s on the punctured faces
    std::optional<Foliation> foliation;

    explicit Session(std::string const& path)
        : weaving_mesh(path)
        , geodesic_field(weaving_mesh)
        , field_operators(assemble_face_field_operators(weaving_mesh))
        , geodesic_optimizer(geodesic_field, field_operators)
    {
    }
};

// Demo presets: the two showcase meshes with the settings verified for
// them (torus: no singularities, full λ sharpening; fertility: smooth start
// and the reference's gentle schedule, otherwise Algorithm 1 crystallizes
// dozens of singularities and Eq. (7) localizes).
struct DemoPreset {
    char const* label;
    char const* filename;
    double lambda_start;
    double lambda_end;
    double shrink;
    double epsilon;
    int max_iterations; // per λ level
    double mu_scale; // Eq. (7) continuation start (× total area)
    double mu_theta; // Eq. (8), × mean face area
    int isolines_per_period;
};

constexpr DemoPreset demo_presets[] = {
    { .label = "torus",
        .filename = "torus.obj",
        .lambda_start = 100.0,
        .lambda_end = 1e-8,
        .shrink = 10.0,
        .epsilon = 1e-6,
        .max_iterations = 50,
        .mu_scale = 1e-4,
        .mu_theta = 0.01,
        .isolines_per_period = 4 },
    { .label = "fertility",
        .filename = "fertility.obj",
        .lambda_start = 100.0,
        .lambda_end = 0.01,
        .shrink = 10.0,
        .epsilon = 1e-6,
        .max_iterations = 10,
        .mu_scale = 1e-4,
        .mu_theta = 0.01,
        .isolines_per_period = 4 },
};
constexpr int demo_preset_count =
    static_cast<int>(std::size(demo_presets));

std::string preset_path(DemoPreset const& preset)
{
    return std::string(DIRECTIONAL_DATA_PATH) + "/" + preset.filename;
}

int selected_preset = 0;
std::unique_ptr<Session> session =
    std::make_unique<Session>(preset_path(demo_presets[0]));
directional::DirectionalViewer viewer;
polyscope::SurfaceFaceScalarQuantity* curl_quantity = nullptr;
bool singularities_enabled = true;
bool curl_enabled = false;
double smoothness = 100.0;
double lambda_end = 1e-8;
double shrink = 10.0;
double epsilon = 1e-6;
int max_iterations = 50;

// Foliation parameters (§4.2.3–4.2.4).
double mu_scale = 1e-4; // μ start for Eq. (7), dimensionless (× total area)
double mu_scale_max = 1.0; // continuation stops here
double mu_theta = 0.01; // μ in Eq. (8), dimensionless (× mean face area)
double scale_multiplier = 1.0; // S in "s ← S·(π/ρ)·s"
int scale_krylov_size = 80; // Lanczos steps for Eq. (7)
int theta_power_iterations = 20; // θ-step inverse iteration
int alternations = 10; // θ/s rounds
int isolines_per_period = 4;

void apply_preset(DemoPreset const& preset)
{
    smoothness = preset.lambda_start;
    lambda_end = preset.lambda_end;
    shrink = preset.shrink;
    epsilon = preset.epsilon;
    max_iterations = preset.max_iterations;
    mu_scale = preset.mu_scale;
    mu_theta = preset.mu_theta;
    isolines_per_period = preset.isolines_per_period;
    // Deterministic smooth start (Knöppel-style field) for both meshes.
    session->geodesic_field.init_smooth();
}

constexpr char const* isolines_name = "Isolines";
constexpr char const* punctured_faces_name = "punctured faces";

void reset_viewer()
{
    polyscope::removeStructure("Mesh 0", false);
    polyscope::removeStructure("Field 0", false);
    polyscope::removeStructure("Singularities 0", false);
    polyscope::removeStructure(isolines_name, false);
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

void load_preset(int index)
{
    DemoPreset const& preset = demo_presets[index];
    std::string const path = preset_path(preset);
    try {
        session = std::make_unique<Session>(path);
    } catch (std::exception const& error) {
        std::println("failed to load {}: {}", path, error.what());
        return;
    }
    selected_preset = index;
    apply_preset(preset);
    reset_viewer();
}

// Scatter a punctured-face quantity back onto the original faces (0 on
// deleted faces).
Eigen::VectorXd faces_to_original(Eigen::VectorXd const& values)
{
    PuncturedMesh const& punctured = *session->punctured;
    Eigen::VectorXd full = Eigen::VectorXd::Zero(
        session->weaving_mesh.mesh().F.rows());
    for (int const face : std::views::iota(0, static_cast<int>(values.size()))) {
        full(punctured.face_to_original(face)) = values(face);
    }
    return full;
}

Eigen::VectorXd vertices_to_original(Eigen::VectorXd const& values)
{
    PuncturedMesh const& punctured = *session->punctured;
    Eigen::VectorXd full = Eigen::VectorXd::Zero(
        session->weaving_mesh.mesh().V.rows());
    for (int const vertex :
        std::views::iota(0, static_cast<int>(values.size()))) {
        full(punctured.vertex_to_original(vertex)) = values(vertex);
    }
    return full;
}

void show_scale()
{
    viewer.set_surface_face_data(faces_to_original(session->scale), "s")
        ->setEnabled(true);
}

void run_puncture()
{
    session->geodesic_field.update_singularities();
    session->punctured =
        puncture(session->weaving_mesh, session->geodesic_field);
    PuncturedMesh const& punctured = *session->punctured;
    session->punctured_operators =
        assemble_face_field_operators(*punctured.sub);
    session->scale.resize(0);
    session->foliation.reset();
    polyscope::removeStructure(isolines_name, false);
    std::println(
        "puncture: {} singular vertices, {} faces deleted, {} faces / {} "
        "vertices remain",
        session->geodesic_field.field().singLocalCycles.size(),
        punctured.deleted_faces.size(),
        punctured.sub->mesh().F.rows(),
        punctured.sub->mesh().V.rows());
    viewer.highlight_faces(punctured.deleted_faces, punctured_faces_name);
}

void run_initial_scale()
{
    if (!session->punctured) {
        run_puncture();
    }
    InitialScaleResult const result = initial_rescaling(
        *session->punctured,
        session->punctured_operators,
        mu_scale,
        mu_scale_max,
        scale_krylov_size,
        1e-10);
    session->scale = result.s;
    session->foliation.reset();
    std::println(
        "initial s: μ {} ({}), {} Lanczos steps, lowest eigenvalues {}, "
        "negative-area fractions {}, chose mode {}",
        result.mu,
        result.sign_consistent ? "sign-consistent" : "NO sign-consistent mode",
        result.iterations,
        result.lowest_eigenvalues.transpose(),
        result.negative_area_fractions.transpose(),
        result.chosen_mode);
    std::println(
        "initial s: rayleigh {}, ‖δ‖_M {}, ‖C(sŵ⊥+δ)‖ {}, s ∈ [{}, {}]",
        result.rayleigh_quotient,
        result.residual_norm,
        result.constraint_norm,
        result.s.minCoeff(),
        result.s.maxCoeff());
    show_scale();
}

void run_rescale()
{
    if (session->scale.size() == 0) {
        run_initial_scale();
    }
    RescaleResult const result =
        global_rescale(*session->punctured, session->scale, scale_multiplier);
    std::println(
        "global rescale: ρ = {}, factor {}, s ∈ [{}, {}]",
        result.rho,
        result.factor,
        session->scale.minCoeff(),
        session->scale.maxCoeff());
    show_scale();
}

void show_foliation()
{
    Foliation const& foliation = *session->foliation;
    FoliationDiagnostics const diagnostics =
        foliation_diagnostics(*session->punctured, foliation);
    std::println(
        "foliation: {} aliased faces, mean alignment error {}",
        diagnostics.aliased_count,
        diagnostics.alignment.mean());

    polyscope::SurfaceVertexScalarQuantity* theta_quantity =
        viewer.set_surface_vertex_data(
            vertices_to_original(foliation.theta), "theta");
    theta_quantity->setColorMap("phase");
    theta_quantity->setMapRange({ -std::numbers::pi, std::numbers::pi });
    theta_quantity->setEnabled(true);
    viewer.set_surface_face_data(
        faces_to_original(diagnostics.alignment), "alignment error");
    show_scale();
}

void run_alternate()
{
    if (session->scale.size() == 0) {
        run_rescale();
    }
    FoliationSolver const solver(*session->punctured, mu_theta);
    session->foliation = solver.alternate(
        session->scale, alternations, theta_power_iterations);
    session->scale = session->foliation->s;
    show_foliation();
}

void run_isolines()
{
    if (!session->foliation) {
        run_alternate();
    }
    IsolineCurves const curves = periodic_isolines(
        session->punctured->sub->mesh(),
        session->foliation->theta,
        isolines_per_period);
    std::println("isolines: {} segments", curves.edges.rows());
    polyscope::removeStructure(isolines_name, false);
    polyscope::CurveNetwork* network = polyscope::registerCurveNetwork(
        isolines_name, curves.nodes, curves.edges);
    network->setRadius(
        0.05 * session->weaving_mesh.mesh().avgEdgeLength, false);
    network->addEdgeScalarQuantity("level", curves.level.cast<double>())
        ->setEnabled(true);
}

template <typename Step>
void guarded(char const* label, Step step)
{
    try {
        step();
    } catch (std::exception const& error) {
        std::println("{} failed: {}", label, error.what());
    }
}

void callback()
{
    if (ImGui::BeginCombo("mesh", demo_presets[selected_preset].label)) {
        for (int const i : std::views::iota(0, demo_preset_count)) {
            bool const selected = i == selected_preset;
            if (ImGui::Selectable(demo_presets[i].label, selected)
                && i != selected_preset) {
                load_preset(i);
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
    if (ImGui::Button("smooth init")) {
        session->geodesic_field.init_smooth();
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

    ImGui::Separator();
    ImGui::TextUnformatted("Foliation (theta from w)");
    ImGui::InputDouble("mu s start (eq. 7)", &mu_scale, 0.0, 0.0, "%.1e");
    ImGui::InputDouble("mu s max (eq. 7)", &mu_scale_max, 0.0, 0.0, "%.1e");
    ImGui::InputDouble("mu theta (eq. 8)", &mu_theta, 0.0, 0.0, "%.1e");
    ImGui::InputDouble("scale multiplier", &scale_multiplier);
    ImGui::InputInt("s krylov size", &scale_krylov_size);
    ImGui::InputInt("theta power iterations", &theta_power_iterations);
    ImGui::InputInt("alternations", &alternations);
    ImGui::InputInt("isolines per period", &isolines_per_period);
    if (ImGui::Button("puncture")) {
        guarded("puncture", run_puncture);
    }
    if (ImGui::Button("initial s")) {
        guarded("initial s", run_initial_scale);
    }
    if (ImGui::Button("rescale s")) {
        guarded("rescale s", run_rescale);
    }
    if (ImGui::Button("alternate theta/s")) {
        guarded("alternate", run_alternate);
    }
    if (ImGui::Button("extract isolines")) {
        guarded("isolines", run_isolines);
    }
    if (ImGui::Button("recover foliation (all)")) {
        guarded("recover foliation", [] {
            run_puncture();
            run_initial_scale();
            run_rescale();
            run_alternate();
            run_isolines();
        });
    }
}

int main()
{
    viewer.init();
    apply_preset(demo_presets[selected_preset]);
    reset_viewer();
    viewer.set_callback(callback);
    viewer.launch();

    return 0;
}
