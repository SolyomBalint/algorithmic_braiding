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
constexpr int demo_preset_count = static_cast<int>(std::size(demo_presets));

std::string preset_path(DemoPreset const& preset)
{
    return std::string(DIRECTIONAL_DATA_PATH) + "/" + preset.filename;
}

int selected_preset = 0;
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
        "negative-area fractions {}, chose mode {}",
        result.mu,
        result.sign_consistent ? "sign-consistent" : "NO sign-consistent mode",
        result.iterations, result.lowest_eigenvalues.transpose(),
        result.negative_area_fractions.transpose(), result.chosen_mode);
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
            mu_scale, mu_scale_max, scale_krylov_size, 1e-10));
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
    IsolineCurves const curves
        = periodic_isolines(session->punctured->sub->mesh(),
            session->foliation->theta, isolines_per_period);
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
        0.02, omega_floor, [](int step, int krylov) {
            if (job.cancel) {
                throw JobCancelled {};
            }
            job.iter = step;
            job.max_iter = krylov;
            pump_ui(false);
        });
    accept_initial_scale(result);
    append_plot(result.rayleigh_quotient);
    bool const done_mu = result.sign_consistent
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
    ImGui::InputInt("isolines per period", &isolines_per_period);
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
