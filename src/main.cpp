#include "face_field_operators.h"
#include "foliation.h"
#include "geodesic_field.h"
#include "geodesic_optimizer.h"
#include "isolines.h"
#include "punctured_mesh.h"
#include "scale_field.h"
#include "weaving_mesh.h"

#include <directional/directional_viewer.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>
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
    bool auto_isoline_density; // derive isolines_per_period from the mesh
    double leaf_spacing_edges; // target leaf spacing, in median-phase edges
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
        .isolines_per_period = 4,
        .auto_isoline_density = false,
        .leaf_spacing_edges = 5.0 },
    { .label = "fertility",
        .filename = "fertility.obj",
        .lambda_start = 100.0,
        .lambda_end = 0.01,
        .shrink = 10.0,
        .epsilon = 1e-6,
        .max_iterations = 10,
        .mu_scale = 1e-4,
        .mu_theta = 0.01,
        .isolines_per_period = 4,
        .auto_isoline_density = false,
        .leaf_spacing_edges = 5.0 },
};
constexpr int demo_preset_count = static_cast<int>(std::size(demo_presets));

std::string preset_path(DemoPreset const& preset)
{
    return std::string(DIRECTIONAL_DATA_PATH) + "/" + preset.filename;
}

int selected_preset = 0;
std::string custom_mesh_name;
char custom_mesh_path[1024] = "";
std::unique_ptr<Session> session
    = std::make_unique<Session>(preset_path(demo_presets[0]));
directional::DirectionalViewer viewer;
polyscope::SurfaceFaceScalarQuantity* curl_quantity = nullptr;
bool singularities_enabled = false;
bool curl_enabled = false;
bool streamlines_enabled = false;
bool streamlines_ready = false;
int streamline_steps = 80;
double streamline_dist_ratio = 1.0;
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
// Leaf density: with the paper's rescale the *largest* edge phase is π, so on
// meshes with a wide edge-length spread a fixed count per period gives leaves
// dozens of edges apart. `auto` derives the count from the median edge phase
// so leaves are ~`leaf_spacing_edges` typical edges apart.
bool auto_isoline_density = true;
double leaf_spacing_edges = 5.0;
double max_localized_area = 0.5; // Eq. (7) mode rejection: low-|s| area
bool leaf_shading = false; // shader contours instead of the curve network

void apply_preset_params(DemoPreset const& preset)
{
    smoothness = preset.lambda_start;
    lambda_end = preset.lambda_end;
    shrink = preset.shrink;
    epsilon = preset.epsilon;
    max_iterations = preset.max_iterations;
    mu_scale = preset.mu_scale;
    mu_theta = preset.mu_theta;
    isolines_per_period = preset.isolines_per_period;
    auto_isoline_density = preset.auto_isoline_density;
    leaf_spacing_edges = preset.leaf_spacing_edges;
}

void apply_preset(DemoPreset const& preset)
{
    apply_preset_params(preset);
    session->geodesic_field.init_smooth();
}

constexpr char const* isolines_name = "Isolines";
constexpr char const* leaf_shading_name = "leaves (theta unwrapped)";
constexpr char const* punctured_faces_name = "punctured faces";
constexpr char const* aliased_faces_name = "aliased faces";
constexpr char const* surface_mesh_name = "Mesh 0"; // DirectionalViewer's
constexpr char const* streamlines_name = "Streamlines 0";

void clear_streamlines()
{
    polyscope::removeStructure(streamlines_name, false);
    streamlines_ready = false;
}

void trace_streamlines()
{
    directional::CartesianField const& field
        = session->geodesic_field.field();
    if (!viewer.slState.empty()) {
        viewer.slState[0] = directional::StreamlineState {};
    }
    viewer.init_streamlines(field, 0, Eigen::VectorXi(),
        std::max(0.1, streamline_dist_ratio));
    int const steps = std::max(1, streamline_steps);
    double const d_time = 0.5 * field.tb->avgAdjLength;
    for (int const i : std::views::iota(0, steps - 1)) {
        directional::streamlines_next(
            viewer.slData[0], viewer.slState[0], d_time);
    }
    viewer.advance_streamlines(0.5);
    viewer.toggle_streamlines(streamlines_enabled);
    streamlines_ready = true;
}

void reset_viewer()
{
    polyscope::removeStructure("Mesh 0", false);
    polyscope::removeStructure("Field 0", false);
    polyscope::removeStructure("Singularities 0", false);
    polyscope::removeStructure(isolines_name, false);
    clear_streamlines();
    curl_quantity = nullptr;
    viewer.set_surface_mesh(session->weaving_mesh.mesh());
    viewer.set_cartesian_field(session->geodesic_field.field());
    viewer.toggle_singularities(singularities_enabled);
    curl_quantity = viewer.set_surface_face_data(
        session->geodesic_field.face_curl(), "curl");
    curl_quantity->setEnabled(curl_enabled);
    if (streamlines_enabled) {
        trace_streamlines();
    }
}

void refresh_viewer()
{
    session->geodesic_field.update_singularities();
    viewer.set_cartesian_field(session->geodesic_field.field());
    viewer.toggle_singularities(singularities_enabled);
    Eigen::VectorXd const curl = session->geodesic_field.face_curl();
    curl_quantity->updateData(curl);
    curl_quantity->setMapRange({ curl.minCoeff(), curl.maxCoeff() });
    if (streamlines_enabled) {
        trace_streamlines();
    } else {
        clear_streamlines();
    }
}

std::string resolve_mesh_path(std::string const& requested)
{
    if (requested.empty()) {
        throw std::runtime_error("empty mesh path");
    }
    std::filesystem::path const path(requested);
    if (std::filesystem::is_regular_file(path)) {
        return std::filesystem::weakly_canonical(path).string();
    }
    std::filesystem::path const in_data
        = std::filesystem::path(DIRECTIONAL_DATA_PATH) / path;
    if (std::filesystem::is_regular_file(in_data)) {
        return std::filesystem::weakly_canonical(in_data).string();
    }
    std::filesystem::path const in_data_name
        = std::filesystem::path(DIRECTIONAL_DATA_PATH) / path.filename();
    if (std::filesystem::is_regular_file(in_data_name)) {
        return std::filesystem::weakly_canonical(in_data_name).string();
    }
    throw std::runtime_error("file not found: " + requested);
}

bool try_load_mesh(std::string const& requested, int preset_index)
{
    std::string path;
    std::unique_ptr<Session> next;
    try {
        path = resolve_mesh_path(requested);
        next = std::make_unique<Session>(path);
        next->geodesic_field.init_smooth();
    } catch (std::exception const& error) {
        std::println("failed to load {}: {}", requested, error.what());
        return false;
    }

    std::unique_ptr<Session> previous = std::move(session);
    session = std::move(next);
    try {
        reset_viewer();
    } catch (std::exception const& error) {
        std::println("failed to display {}: {}", path, error.what());
        session = std::move(previous);
        try {
            reset_viewer();
        } catch (std::exception const& restore_error) {
            std::println(
                "failed to restore previous mesh: {}", restore_error.what());
        }
        return false;
    }

    if (preset_index >= 0 && preset_index < demo_preset_count) {
        selected_preset = preset_index;
        apply_preset_params(demo_presets[preset_index]);
    } else {
        selected_preset = -1;
        custom_mesh_name = std::filesystem::path(path).filename().string();
        // Unknown mesh: let the leaf density follow its edge lengths.
        auto_isoline_density = true;
    }
    return true;
}

void load_preset(int index)
{
    DemoPreset const& preset = demo_presets[index];
    (void)try_load_mesh(preset_path(preset), index);
}

std::vector<std::string> const& data_mesh_files()
{
    static std::vector<std::string> files;
    static bool scanned = false;
    if (scanned) {
        return files;
    }
    scanned = true;
    try {
        for (std::filesystem::directory_entry const& entry :
            std::filesystem::directory_iterator(DIRECTIONAL_DATA_PATH)) {
            try {
                if (!entry.is_regular_file()) {
                    continue;
                }
            } catch (std::exception const&) {
                continue;
            }
            std::string const name = entry.path().filename().string();
            std::string const ext = mesh_extension(name);
            if (ext == ".obj" || ext == ".off") {
                files.push_back(name);
            }
        }
        std::ranges::sort(files);
    } catch (std::exception const& error) {
        std::println(
            "cannot list {}: {}", DIRECTIONAL_DATA_PATH, error.what());
    }
    return files;
}

// Scatter a punctured-face quantity back onto the original faces (0 on
// deleted faces).
Eigen::VectorXd faces_to_original(Eigen::VectorXd const& values)
{
    PuncturedMesh const& punctured = *session->punctured;
    Eigen::VectorXd full
        = Eigen::VectorXd::Zero(session->weaving_mesh.mesh().F.rows());
    for (int const face :
        std::views::iota(0, static_cast<int>(values.size()))) {
        full(punctured.face_to_original(face)) = values(face);
    }
    return full;
}

Eigen::VectorXd vertices_to_original(Eigen::VectorXd const& values)
{
    PuncturedMesh const& punctured = *session->punctured;
    Eigen::VectorXd full
        = Eigen::VectorXd::Zero(session->weaving_mesh.mesh().V.rows());
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
    session->punctured
        = puncture(session->weaving_mesh, session->geodesic_field);
    PuncturedMesh const& punctured = *session->punctured;
    session->punctured_operators
        = assemble_face_field_operators(*punctured.sub);
    session->scale.resize(0);
    session->foliation.reset();
    polyscope::removeStructure(isolines_name, false);
    polyscope::getSurfaceMesh(surface_mesh_name)
        ->removeQuantity(leaf_shading_name, false);
    polyscope::getSurfaceMesh(surface_mesh_name)
        ->removeQuantity(aliased_faces_name, false);
    std::println(
        "puncture: {} singular vertices, {} faces deleted, {} faces / {} "
        "vertices remain",
        session->geodesic_field.field().singLocalCycles.size(),
        punctured.deleted_faces.size(), punctured.sub->mesh().F.rows(),
        punctured.sub->mesh().V.rows());
    viewer.highlight_faces(punctured.deleted_faces, punctured_faces_name);
}

void accept_initial_scale(InitialScaleResult const& result)
{
    session->scale = result.s;
    session->foliation.reset();
    std::println(
        "initial s: μ {} ({}), {} Lanczos steps, lowest eigenvalues {}, "
        "negative-area fractions {}, low-|s| area fractions {}, chose mode {}",
        result.mu,
        result.accepted ? "accepted"
                        : "NO sign-consistent, non-localized mode",
        result.iterations, result.lowest_eigenvalues.transpose(),
        result.negative_area_fractions.transpose(),
        result.low_scale_area_fractions.transpose(), result.chosen_mode);
    std::println(
        "initial s: rayleigh {}, ‖δ‖_M {}, ‖C(sŵ⊥+δ)‖ {}, s ∈ [{}, {}]",
        result.rayleigh_quotient, result.residual_norm, result.constraint_norm,
        result.s.minCoeff(), result.s.maxCoeff());
    show_scale();
}

void run_initial_scale()
{
    if (!session->punctured) {
        run_puncture();
    }
    accept_initial_scale(
        initial_rescaling(*session->punctured, session->punctured_operators,
            mu_scale, mu_scale_max, scale_krylov_size, 1e-10, 0.02,
            max_localized_area));
}

void run_rescale()
{
    if (session->scale.size() == 0) {
        run_initial_scale();
    }
    RescaleResult const result
        = global_rescale(*session->punctured, session->scale, scale_multiplier);
    std::println("global rescale: ρ = {}, factor {}, s ∈ [{}, {}]", result.rho,
        result.factor, session->scale.minCoeff(), session->scale.maxCoeff());
    show_scale();
}

void show_foliation()
{
    Foliation const& foliation = *session->foliation;
    FoliationDiagnostics const diagnostics
        = foliation_diagnostics(*session->punctured, foliation);
    std::println("foliation: {} aliased faces, mean alignment error {}",
        diagnostics.aliased_count, diagnostics.alignment.mean());

    polyscope::SurfaceVertexScalarQuantity* theta_quantity
        = viewer.set_surface_vertex_data(
            vertices_to_original(foliation.theta), "theta");
    theta_quantity->setColorMap("phase");
    theta_quantity->setMapRange({ -std::numbers::pi, std::numbers::pi });
    theta_quantity->setEnabled(true);
    viewer.set_surface_face_data(
        faces_to_original(diagnostics.alignment), "alignment error");
    // Faces the isoline extractor will skip (θ winds inside them): kept as
    // a disabled colour quantity so gaps in the leaves can be explained.
    Eigen::VectorXi aliased(diagnostics.aliased_count);
    int next = 0;
    for (int const face :
        std::views::iota(0, static_cast<int>(diagnostics.aliased.size()))) {
        if (diagnostics.aliased(face)) {
            aliased(next++) = session->punctured->face_to_original(face);
        }
    }
    viewer.highlight_faces(aliased, aliased_faces_name)->setEnabled(false);
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

// Shader-side leaves: a corner scalar quantity holding the per-face
// unwrapped θ (see unwrapped_corner_theta) with Polyscope's isoline
// contouring at period `spacing`. Screen-space line width, so the leaves
// stay visible at any zoom and mesh scale, and no ±π seam.
void show_leaf_shading(double spacing)
{
    PuncturedMesh const& punctured = *session->punctured;
    Eigen::MatrixXi const& sub_faces = punctured.sub->mesh().F;
    Eigen::VectorXd const sub_corners = unwrapped_corner_theta(
        sub_faces, session->foliation->theta, spacing);
    Eigen::VectorXd corners
        = Eigen::VectorXd::Zero(3 * session->weaving_mesh.mesh().F.rows());
    for (int const face :
        std::views::iota(0, static_cast<int>(sub_faces.rows()))) {
        corners.segment<3>(3 * punctured.face_to_original(face))
            = sub_corners.segment<3>(3 * face);
    }
    polyscope::SurfaceCornerScalarQuantity* quantity
        = polyscope::getSurfaceMesh(surface_mesh_name)
              ->addCornerScalarQuantity(leaf_shading_name, corners);
    // A huge map range gives a uniform base colour; only the contours show.
    quantity->setColorMap("blues");
    quantity->setMapRange({ -1e6, 1e6 });
    quantity->setIsolinesEnabled(true);
    quantity->setIsolineStyle(polyscope::IsolineStyle::Contour);
    quantity->setIsolinePeriod(spacing, false);
    quantity->setIsolineContourThickness(0.15);
    quantity->setIsolineDarkness(0.9); // line strength, 1 = black
    quantity->setEnabled(leaf_shading);
}

void run_isolines()
{
    if (!session->foliation) {
        run_alternate();
    }
    PuncturedMesh const& punctured = *session->punctured;
    Foliation const& foliation = *session->foliation;

    if (auto_isoline_density) {
        isolines_per_period = auto_isolines_per_period(
            punctured, foliation.s, leaf_spacing_edges);
    }
    isolines_per_period = std::max(1, isolines_per_period);
    double const spacing = 2.0 * std::numbers::pi / isolines_per_period;

    // Which of the three "few leaves" causes applies: edge phases near π
    // everywhere means the mesh is at the aliasing bound (raise the count),
    // a small median against the max means a wide edge-length or s spread,
    // a large low-|s| area means Eq. (7) returned a localized mode.
    Eigen::VectorXd const phases = edge_phases(punctured, foliation.s);
    MeshTables const tables = mesh_tables(punctured.sub->mesh());
    std::println(
        "isolines: edge phase p50 {:.3g}, p90 {:.3g}, max {:.3g} (π = {:.3g}); "
        "low-|s| area {:.1f}%; {} levels per period ({})",
        quantile(phases, 0.5), quantile(phases, 0.9),
        phases.size() > 0 ? phases.maxCoeff() : 0.0, std::numbers::pi,
        100.0 * low_scale_area_fraction(tables.face_areas, foliation.s),
        isolines_per_period, auto_isoline_density ? "auto" : "manual");

    IsolineCurves const curves = periodic_isolines(
        punctured.sub->mesh(), foliation.theta, isolines_per_period);
    Eigen::VectorXi per_level = Eigen::VectorXi::Zero(isolines_per_period);
    for (int const i :
        std::views::iota(0, static_cast<int>(curves.level.size()))) {
        ++per_level(curves.level(i));
    }
    std::println("isolines: {} segments, per level min {} max {}",
        curves.edges.rows(), per_level.minCoeff(), per_level.maxCoeff());

    polyscope::removeStructure(isolines_name, false);
    polyscope::CurveNetwork* network = polyscope::registerCurveNetwork(
        isolines_name, curves.nodes, curves.edges);
    // Relative radius: a fraction of the scene scale rather than of the
    // average edge, which is sub-pixel on large, finely meshed models.
    network->setRadius(0.0015, true);
    network->setMaterial("flat");
    network->addEdgeScalarQuantity("level", curves.level.cast<double>())
        ->setColorMap("turbo")
        ->setEnabled(true);
    network->setEnabled(!leaf_shading);
    show_leaf_shading(spacing);
}

void toggle_leaf_shading()
{
    leaf_shading = !leaf_shading;
    if (polyscope::hasCurveNetwork(isolines_name)) {
        polyscope::getCurveNetwork(isolines_name)->setEnabled(!leaf_shading);
    }
    polyscope::Quantity* quantity
        = polyscope::getSurfaceMesh(surface_mesh_name)
              ->getQuantity(leaf_shading_name);
    if (quantity != nullptr) {
        quantity->setEnabled(leaf_shading);
    }
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

struct JobCancelled : std::runtime_error {
    JobCancelled()
        : std::runtime_error("cancelled")
    {
    }
};

enum class JobKind { Idle, Optimize, Recover };
enum class RecoverStage { Puncture, InitialS, Rescale, Alternate, Isolines };

struct JobState {
    JobKind kind = JobKind::Idle;
    bool cancel = false;
    RecoverStage recover_stage = RecoverStage::Puncture;
    RecoverStage recover_until = RecoverStage::Isolines;
    std::string stage;
    std::string plot_label = "energy";
    std::vector<float> plot;
    float last_value = 0.0f;
    double lambda = 0.0;
    double mu = 0.0;
    double current_lambda = 0.0;
    double current_mu = 0.0;
    int iter = 0;
    int max_iter = 0;
    int level_iter = 0;
    bool unconstrained_done = false;
    bool enforce_curl = false;
};

JobState job;
std::chrono::steady_clock::time_point last_ui_pump {};

void append_plot(double value)
{
    constexpr int cap = 512;
    job.plot.push_back(static_cast<float>(value));
    if (static_cast<int>(job.plot.size()) > cap) {
        job.plot.erase(job.plot.begin(),
            job.plot.begin() + static_cast<int>(job.plot.size()) - cap);
    }
    job.last_value = static_cast<float>(value);
}

void pump_ui(bool force)
{
    using clock = std::chrono::steady_clock;
    auto const now = clock::now();
    if (!force && last_ui_pump != clock::time_point {}
        && now - last_ui_pump < std::chrono::milliseconds(300)) {
        return;
    }
    last_ui_pump = now;
    if (polyscope::windowRequestsClose()) {
        job.cancel = true;
    }
    auto const saved_limit = polyscope::options::frameTickLimitFPSMode;
    polyscope::options::frameTickLimitFPSMode
        = polyscope::LimitFPSMode::IgnoreLimits;
    polyscope::requestRedraw();
    polyscope::frameTick();
    polyscope::options::frameTickLimitFPSMode = saved_limit;
}

void finish_job()
{
    job.kind = JobKind::Idle;
    job.cancel = false;
    job.stage.clear();
    polyscope::requestRedraw();
}

void start_optimize()
{
    if (shrink <= 1.0 || smoothness < lambda_end) {
        std::println("optimize: need shrink > 1 and lambda >= lambda end");
        return;
    }
    job = JobState {};
    job.kind = JobKind::Optimize;
    job.stage = "optimize";
    job.plot_label = "energy";
    job.current_lambda = smoothness;
    job.max_iter = max_iterations;
}

void start_recover(RecoverStage from, RecoverStage until)
{
    job = JobState {};
    job.kind = JobKind::Recover;
    job.recover_stage = from;
    job.recover_until = until;
    job.current_mu = mu_scale;
    job.stage = "recover";
    job.plot_label = "energy";
}

void tick_optimize()
{
    if (job.cancel) {
        throw JobCancelled {};
    }
    double const lambda = job.current_lambda;
    bool const curl = job.unconstrained_done;
    job.lambda = lambda;
    job.enforce_curl = curl;
    job.stage = curl ? "optimize" : "optimize (smooth start)";
    job.max_iter = max_iterations;

    double const change = curl
        ? session->geodesic_optimizer.step(lambda)
        : session->geodesic_optimizer.step_unconstrained(lambda);
    job.level_iter += 1;
    job.iter = job.level_iter;
    append_plot(session->geodesic_optimizer.energy(lambda).total);

    bool const level_done = job.level_iter >= max_iterations
        || change < session->geodesic_optimizer.mass_norm_threshold(epsilon);
    if (!level_done) {
        return;
    }

    std::println("λ={}  {}  level done: {} iterations", lambda,
        curl ? "C" : "no-C", job.level_iter);
    refresh_viewer();

    if (!job.unconstrained_done) {
        job.unconstrained_done = true;
        job.level_iter = 0;
        return;
    }
    if (lambda <= lambda_end) {
        finish_job();
        return;
    }
    job.current_lambda = std::max(lambda / shrink, lambda_end);
    job.level_iter = 0;
}

void tick_initial_s()
{
    if (!session->punctured) {
        job.recover_stage = RecoverStage::Puncture;
        return;
    }
    job.stage = "initial s";
    job.plot_label = "rayleigh";
    job.mu = job.current_mu;
    job.lambda = 0.0;
    InitialScaleResult const result = initial_rescaling_at(*session->punctured,
        session->punctured_operators, job.current_mu, scale_krylov_size, 1e-10,
        0.02, max_localized_area, omega_floor, [](int step, int krylov) {
            if (job.cancel) {
                throw JobCancelled {};
            }
            job.iter = step;
            job.max_iter = krylov;
            pump_ui(false);
        });
    accept_initial_scale(result);
    append_plot(result.rayleigh_quotient);
    bool const done_mu = result.accepted
        || job.current_mu >= mu_scale_max * (1.0 + 1e-12);
    if (!done_mu) {
        job.current_mu *= 10.0;
        return;
    }
    if (job.recover_until == RecoverStage::InitialS) {
        finish_job();
        return;
    }
    job.recover_stage = RecoverStage::Rescale;
}

void tick_alternate()
{
    if (session->scale.size() == 0) {
        job.recover_stage = RecoverStage::Rescale;
        return;
    }
    job.stage = "alternate";
    job.plot_label = "energy(8)";
    job.max_iter = alternations;
    FoliationSolver const solver(*session->punctured, mu_theta);
    session->foliation
        = solver.alternate(session->scale, alternations, theta_power_iterations,
            [](int round, int total, Foliation const& foliation,
                double energy_value, double) {
                if (job.cancel) {
                    throw JobCancelled {};
                }
                job.iter = round;
                job.max_iter = total;
                session->foliation = foliation;
                session->scale = foliation.s;
                show_foliation();
                append_plot(energy_value);
                pump_ui(true);
            });
    session->scale = session->foliation->s;
    if (job.recover_until == RecoverStage::Alternate) {
        finish_job();
        return;
    }
    job.recover_stage = RecoverStage::Isolines;
}

void tick_recover()
{
    if (job.cancel) {
        throw JobCancelled {};
    }
    switch (job.recover_stage) {
    case RecoverStage::Puncture:
        job.stage = "puncture";
        run_puncture();
        job.current_mu = mu_scale;
        if (job.recover_until == RecoverStage::Puncture) {
            finish_job();
            return;
        }
        job.recover_stage = RecoverStage::InitialS;
        break;
    case RecoverStage::InitialS:
        tick_initial_s();
        break;
    case RecoverStage::Rescale:
        job.stage = "rescale s";
        run_rescale();
        if (job.recover_until == RecoverStage::Rescale) {
            finish_job();
            return;
        }
        job.recover_stage = RecoverStage::Alternate;
        break;
    case RecoverStage::Alternate:
        tick_alternate();
        break;
    case RecoverStage::Isolines:
        job.stage = "isolines";
        run_isolines();
        finish_job();
        break;
    }
}

void pump_job()
{
    if (job.kind == JobKind::Idle) {
        return;
    }
    try {
        if (job.kind == JobKind::Optimize) {
            tick_optimize();
        } else {
            tick_recover();
        }
    } catch (JobCancelled const&) {
        std::println("{} cancelled", job.stage);
        finish_job();
    } catch (std::exception const& error) {
        std::println("{} failed: {}", job.stage, error.what());
        finish_job();
    }
}

void draw_job_ui()
{
    if (job.kind == JobKind::Idle) {
        return;
    }
    ImGui::Separator();
    char const spin[] = "|/-\\";
    int const frame = static_cast<int>(ImGui::GetTime() * 8.0) % 4;
    ImGui::Text("%c  %s", spin[frame], job.stage.c_str());
    if (job.kind == JobKind::Optimize) {
        ImGui::Text("λ=%g  %s  it=%d/%d", job.lambda,
            job.enforce_curl ? "C" : "no-C", job.iter, job.max_iter);
    } else if (job.mu != 0.0) {
        ImGui::Text("μ=%g  it=%d/%d", job.mu, job.iter, job.max_iter);
    } else if (job.max_iter > 0) {
        ImGui::Text("it=%d/%d", job.iter, job.max_iter);
    }
    if (job.max_iter > 0) {
        ImGui::ProgressBar(
            static_cast<float>(job.iter) / static_cast<float>(job.max_iter),
            ImVec2(-1.0f, 0.0f));
    }
    if (!job.plot.empty()) {
        ImGui::PlotLines("##jobplot", job.plot.data(),
            static_cast<int>(job.plot.size()), 0, job.plot_label.c_str(),
            FLT_MAX, FLT_MAX, ImVec2(0.0f, 64.0f));
        ImGui::Text("%s = %g", job.plot_label.c_str(), job.last_value);
    }
    if (ImGui::Button("cancel")) {
        job.cancel = true;
    }
}

void callback()
{
    draw_job_ui();
    bool const busy = job.kind != JobKind::Idle;
    ImGui::BeginDisabled(busy);

    char const* const mesh_label = selected_preset >= 0
        ? demo_presets[selected_preset].label
        : (custom_mesh_name.empty() ? "custom" : custom_mesh_name.c_str());
    if (ImGui::BeginCombo("mesh", mesh_label)) {
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
        ImGui::Separator();
        for (std::string const& name : data_mesh_files()) {
            if (name == "torus.obj" || name == "fertility.obj") {
                continue;
            }
            bool const selected
                = selected_preset < 0 && custom_mesh_name == name;
            if (ImGui::Selectable(name.c_str(), selected)) {
                (void)try_load_mesh(
                    std::string(DIRECTIONAL_DATA_PATH) + "/" + name, -1);
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::CollapsingHeader("load mesh")) {
        ImGui::InputText("path", custom_mesh_path, sizeof custom_mesh_path);
        if (ImGui::Button("load path")) {
            (void)try_load_mesh(custom_mesh_path, -1);
        }
    }
    ImGui::InputDouble("lambda", &smoothness);
    ImGui::InputDouble("lambda end", &lambda_end);
    ImGui::InputDouble("shrink", &shrink);
    ImGui::InputDouble("epsilon", &epsilon);
    ImGui::InputInt("max iterations", &max_iterations);
    if (ImGui::Button("optimize")) {
        start_optimize();
    }
    if (ImGui::Button("toggle curl map")) {
        curl_enabled = !curl_quantity->isEnabled();
        curl_quantity->setEnabled(curl_enabled);
    }
    if (ImGui::Button("toggle singularities")) {
        singularities_enabled = !singularities_enabled;
        viewer.toggle_singularities(singularities_enabled);
    }
    ImGui::InputInt("streamline steps", &streamline_steps);
    ImGui::InputDouble("streamline spacing", &streamline_dist_ratio);
    if (ImGui::Button("toggle streamlines")) {
        streamlines_enabled = !streamlines_enabled;
        if (streamlines_enabled) {
            trace_streamlines();
        } else if (streamlines_ready) {
            viewer.toggle_streamlines(false);
        }
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
    ImGui::InputDouble(
        "max low-s area (eq. 7)", &max_localized_area, 0.0, 0.0, "%.2f");
    ImGui::Checkbox("auto isolines per period", &auto_isoline_density);
    ImGui::InputDouble("leaf spacing (edges)", &leaf_spacing_edges);
    ImGui::BeginDisabled(auto_isoline_density);
    ImGui::InputInt("isolines per period", &isolines_per_period);
    ImGui::EndDisabled();
    if (ImGui::Button(leaf_shading ? "leaves: shader contours (toggle)"
                                   : "leaves: curve network (toggle)")) {
        toggle_leaf_shading();
    }
    if (ImGui::Button("puncture")) {
        guarded("puncture", run_puncture);
    }
    if (ImGui::Button("initial s")) {
        RecoverStage const from = session->punctured ? RecoverStage::InitialS
                                                     : RecoverStage::Puncture;
        start_recover(from, RecoverStage::InitialS);
    }
    if (ImGui::Button("rescale s")) {
        guarded("rescale s", run_rescale);
    }
    if (ImGui::Button("alternate theta/s")) {
        RecoverStage from = RecoverStage::Alternate;
        if (!session->punctured) {
            from = RecoverStage::Puncture;
        } else if (session->scale.size() == 0) {
            from = RecoverStage::InitialS;
        }
        start_recover(from, RecoverStage::Alternate);
    }
    if (ImGui::Button("extract isolines")) {
        guarded("isolines", run_isolines);
    }
    if (ImGui::Button("recover foliation (all)")) {
        start_recover(RecoverStage::Puncture, RecoverStage::Isolines);
    }

    ImGui::EndDisabled();
}

int main()
{
    viewer.init();
    apply_preset(demo_presets[selected_preset]);
    reset_viewer();
    viewer.set_callback(callback);
    while (!polyscope::windowRequestsClose()) {
        pump_job();
        if (polyscope::windowRequestsClose()) {
            break;
        }
        polyscope::frameTick();
    }

    return 0;
}
