#pragma once

#include "face_field_operators.h"
#include "punctured_mesh.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cmath>
#include <numbers>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

// ŵ⊥ = n × ŵ per face, in ambient coordinates (|F| × 3).
inline Eigen::MatrixXd perpendicular_ambient(PuncturedMesh const& punctured)
{
    directional::TriMesh const& mesh = punctured.sub->mesh();
    int const face_count = mesh.F.rows();
    Eigen::MatrixXd perp(face_count, 3);
    for (int const face : std::views::iota(0, face_count)) {
        Eigen::RowVector3d const normal = mesh.faceNormals.row(face);
        Eigen::RowVector3d const w = punctured.field.extField.row(face);
        perp.row(face) = normal.cross(w);
    }
    return perp;
}

// The same ŵ⊥ expressed in each face's orthonormal frame (FBx, FBy),
// (|F| × 2), so that it is consistent with the ambient version above.
inline Eigen::MatrixXd perpendicular_intrinsic(PuncturedMesh const& punctured)
{
    directional::TriMesh const& mesh = punctured.sub->mesh();
    Eigen::MatrixXd const ambient = perpendicular_ambient(punctured);
    Eigen::MatrixXd perp(ambient.rows(), 2);
    for (int const face : std::views::iota(0, static_cast<int>(ambient.rows()))) {
        perp(face, 0) = ambient.row(face).dot(mesh.FBx.row(face));
        perp(face, 1) = ambient.row(face).dot(mesh.FBy.row(face));
    }
    return perp;
}

// The matrices of Eq. (7) in the unknown x = [δ; s] ∈ R^{3|F|}:
//   constraint  B  = [C, C S]           (|E_int| × 3|F|),
//   energy      A₁ = diag(M, μ A Q_s),
//   norm        A₂ = diag(M, M_F),
// where S stacks ŵ_f⊥ so that S s = (s_f ŵ_f⊥)_f and A is the total surface
// area. The δ and s norms carry a factor of area while sᵀQ_s s does not, so
// the paper's absolute μ (1e-4 on its meshes) is mesh-scale dependent; here
// μ is dimensionless. With μA the cost of letting s vary over the whole
// surface is ~μ, and a mode localized to a patch of diameter L costs
// ~μ (diam/L)², which is what keeps Eq. (7) from returning a localized
// bump on surfaces with many singularities (fertility needs μ ≈ 1e-2;
// 1e-2 also reproduces the sphere and torus results).
struct ScaleSystem {
    Eigen::SparseMatrix<double> constraint;
    Eigen::SparseMatrix<double> energy;
    Eigen::SparseMatrix<double> norm;
    int face_count = 0;
};

inline ScaleSystem assemble_scale_system(
    PuncturedMesh const& punctured,
    FaceFieldOperators const& operators,
    double mu,
    double scalar_weight_floor = omega_floor)
{
    MeshTables const tables = mesh_tables(punctured.sub->mesh());
    int const face_count = tables.faces.rows();
    int const stacked_size = 2 * face_count;
    int const unknown_count = stacked_size + face_count;

    Eigen::MatrixXd const perp = perpendicular_intrinsic(punctured);

    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(2 * face_count);
    for (int const face : std::views::iota(0, face_count)) {
        entries.emplace_back(2 * face, face, perp(face, 0));
        entries.emplace_back(2 * face + 1, face, perp(face, 1));
    }
    Eigen::SparseMatrix<double> perp_placement(stacked_size, face_count);
    perp_placement.setFromTriplets(entries.begin(), entries.end());

    Eigen::SparseMatrix<double> const& curl = operators.curl_matrix;
    Eigen::SparseMatrix<double> const curl_perp = curl * perp_placement;

    ScaleSystem system;
    system.face_count = face_count;

    entries.clear();
    entries.reserve(curl.nonZeros() + curl_perp.nonZeros());
    for (int col = 0; col < curl.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(curl, col); it;
             ++it) {
            entries.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int col = 0; col < curl_perp.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(curl_perp, col); it;
             ++it) {
            entries.emplace_back(it.row(), stacked_size + it.col(), it.value());
        }
    }
    system.constraint.resize(curl.rows(), unknown_count);
    system.constraint.setFromTriplets(entries.begin(), entries.end());

    Eigen::SparseMatrix<double> const scalar_laplacian =
        face_scalar_laplacian(tables, scalar_weight_floor);
    std::vector<Eigen::Triplet<double>> energy_entries;
    std::vector<Eigen::Triplet<double>> norm_entries;
    Eigen::SparseMatrix<double> const& mass = operators.mass_matrix;
    for (int col = 0; col < mass.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mass, col); it;
             ++it) {
            energy_entries.emplace_back(it.row(), it.col(), it.value());
            norm_entries.emplace_back(it.row(), it.col(), it.value());
        }
    }
    double const mu_scaled = mu * tables.face_areas.sum();
    for (int col = 0; col < scalar_laplacian.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(
                 scalar_laplacian, col);
             it;
             ++it) {
            energy_entries.emplace_back(
                stacked_size + it.row(),
                stacked_size + it.col(),
                mu_scaled * it.value());
        }
    }
    for (int const face : std::views::iota(0, face_count)) {
        norm_entries.emplace_back(
            stacked_size + face, stacked_size + face, tables.face_areas(face));
    }
    system.energy.resize(unknown_count, unknown_count);
    system.energy.setFromTriplets(energy_entries.begin(), energy_entries.end());
    system.norm.resize(unknown_count, unknown_count);
    system.norm.setFromTriplets(norm_entries.begin(), norm_entries.end());
    return system;
}

struct InitialScaleResult {
    Eigen::VectorXd s;
    double rayleigh_quotient = 0.0;
    Eigen::VectorXd lowest_eigenvalues; // a few Ritz values, ascending
    Eigen::VectorXd negative_area_fractions; // per Ritz mode, same order
    int chosen_mode = 0; // index into the two vectors above
    double negative_area_fraction = 0.0; // of the chosen s
    bool sign_consistent = false; // chosen mode passed the sign test
    double mu = 0.0; // μ the result was computed with
    double residual_norm = 0.0; // ‖δ‖_M
    double constraint_norm = 0.0; // ‖B x‖ = ‖C(s ŵ⊥ + δ)‖
    int iterations = 0;
};

// §4.2.3 / App. C, Eq. (7): the rescaling s that makes s ŵ⊥ + δ curl-free
// with the smallest ‖δ‖² + μ‖∇s‖², normalized by ‖s‖² + ‖δ‖² = 1:
//     min xᵀ A₁ x / xᵀ A₂ x   s.t.  B x = 0.
// The paper's Algorithm 2 is inverse power iteration on this pencil. We keep
// its building block, the shift-invert operator
//     T x = x̃,   [ A₁+σA₂   Bᵀ ] [x̃]   [ A₂ x ]
//                [ B       −εI ] [μ ] = [  0   ]
// (the exact inverse iteration on null(B): x̃ minimizes ½x̃ᵀ(A₁+σA₂)x̃ − x̃ᵀA₂x
// subject to Bx̃ = 0; the paper's unconstrained solve followed by a
// projection only approximates it), but drive it with Lanczos in the
// A₂-inner product instead of plain power iteration: the lowest eigenvalues
// of Eq. (7) are often nearly degenerate (on a torus every integrable
// rescaling g(ψ)/(R + r cos φ) is a candidate, separated only by μ‖∇s‖²),
// where power iteration needs thousands of solves and stalls on a mixture.
// T is A₂-self-adjoint on null(B), and Krylov vectors started in null(B)
// stay there, so the Ritz values of the tridiagonal are 1/(λ+σ) for the
// constrained eigenvalues λ; the largest Ritz pair gives the minimizer.
//
// Mode selection: θ must be a submersion (§3), i.e. ∇θ = s ŵ⊥ ≠ 0, so s
// must not change sign. Eq. (7) does not know that, and on surfaces with
// singularities its lowest mode is often a sign-changing one (θ ≈ sin 2λ
// around a pole instead of θ = λ: same leaves, but bunched and with a
// critical set). We therefore take the lowest Ritz mode whose s has at
// most `max_negative_area` of the area negative (after the sign flip),
// falling back to the lowest mode when none qualifies.
inline InitialScaleResult initial_rescaling_at(
    PuncturedMesh const& punctured,
    FaceFieldOperators const& operators,
    double mu,
    int krylov_size,
    double tolerance,
    double max_negative_area = 0.02,
    double scalar_weight_floor = omega_floor)
{
    MeshTables const tables = mesh_tables(punctured.sub->mesh());
    ScaleSystem const system =
        assemble_scale_system(punctured, operators, mu, scalar_weight_floor);
    int const face_count = system.face_count;
    int const stacked_size = 2 * face_count;
    int const unknown_count = stacked_size + face_count;
    Eigen::SparseMatrix<double> const& constraint = system.constraint;
    Eigen::SparseMatrix<double> const& energy = system.energy;
    Eigen::SparseMatrix<double> const& norm = system.norm;
    Eigen::SparseMatrix<double> const& mass = operators.mass_matrix;

    // Spectral shift: keeps A₁+σA₂ SPD without moving eigenvectors. Small
    // against the eigenvalues (~1e-2), scale-free.
    double const shift = 1e-6 * energy.diagonal().sum() / norm.diagonal().sum();
    Eigen::SparseMatrix<double> const hessian = energy + shift * norm;

    // Same scaling rule as GeodesicOptimizer::factor_kkt: ε follows the
    // mass scale, so the constraint holds to ~1e-10 relative.
    constexpr double relative_regularization = 1e-10;
    double const regularization = relative_regularization
        * tables.face_areas.sum() / static_cast<double>(face_count);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(
        assemble_quasi_definite_kkt(hessian, constraint, regularization));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("initial_rescaling: KKT LDLT failed");
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(unknown_count + constraint.rows());
    auto apply_operator = [&](Eigen::VectorXd const& x) {
        rhs.head(unknown_count) = norm * x;
        Eigen::VectorXd const solution = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("initial_rescaling: KKT solve failed");
        }
        return Eigen::VectorXd(solution.head(unknown_count));
    };
    auto inner = [&](Eigen::VectorXd const& a, Eigen::VectorXd const& b) {
        return a.dot(norm * b);
    };

    // Lanczos with full reorthogonalization (A₂-inner product).
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::VectorXd start(unknown_count);
    for (int const i : std::views::iota(0, unknown_count)) {
        start(i) = dist(rng);
    }
    // One application projects the random start into null(B).
    Eigen::VectorXd q = apply_operator(start);
    q /= std::sqrt(inner(q, q));

    std::vector<Eigen::VectorXd> basis;
    std::vector<double> alpha;
    std::vector<double> beta;
    basis.push_back(q);
    int steps = 0;
    for (int const j : std::views::iota(0, krylov_size)) {
        Eigen::VectorXd w = apply_operator(basis[j]);
        double const a = inner(basis[j], w);
        alpha.push_back(a);
        w -= a * basis[j];
        if (j > 0) {
            w -= beta[j - 1] * basis[j - 1];
        }
        for (Eigen::VectorXd const& v : basis) {
            w -= inner(v, w) * v;
        }
        double const b = std::sqrt(std::max(inner(w, w), 0.0));
        steps = j + 1;
        if (b <= tolerance * std::abs(a)) {
            break; // invariant subspace found
        }
        beta.push_back(b);
        basis.push_back(w / b);
    }

    Eigen::MatrixXd tridiagonal = Eigen::MatrixXd::Zero(steps, steps);
    for (int const j : std::views::iota(0, steps)) {
        tridiagonal(j, j) = alpha[j];
        if (j + 1 < steps) {
            tridiagonal(j, j + 1) = beta[j];
            tridiagonal(j + 1, j) = beta[j];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ritz(tridiagonal);
    // Eigenvalues of T are 1/(λ+σ), largest first in the order below.
    auto ritz_vector = [&](int rank) {
        int const column = steps - 1 - rank;
        Eigen::VectorXd x = Eigen::VectorXd::Zero(unknown_count);
        for (int const j : std::views::iota(0, steps)) {
            x += ritz.eigenvectors()(j, column) * basis[j];
        }
        x /= std::sqrt(inner(x, x));
        // Eigenvectors are defined up to sign: take the branch with
        // positive total rescaling.
        if (x.tail(face_count).dot(tables.face_areas) < 0.0) {
            x = -x;
        }
        return x;
    };
    auto negative_area = [&](Eigen::VectorXd const& x) {
        double negative = 0.0;
        for (int const face : std::views::iota(0, face_count)) {
            if (x(stacked_size + face) < 0.0) {
                negative += tables.face_areas(face);
            }
        }
        return negative / tables.face_areas.sum();
    };

    InitialScaleResult result;
    result.iterations = steps;
    int const candidates = std::min(steps, 8);
    result.lowest_eigenvalues.resize(candidates);
    result.negative_area_fractions.resize(candidates);
    std::vector<Eigen::VectorXd> vectors;
    for (int const k : std::views::iota(0, candidates)) {
        result.lowest_eigenvalues(k) =
            1.0 / ritz.eigenvalues()(steps - 1 - k) - shift;
        vectors.push_back(ritz_vector(k));
        result.negative_area_fractions(k) = negative_area(vectors[k]);
    }
    result.mu = mu;
    result.chosen_mode = 0;
    result.sign_consistent = false;
    for (int const k : std::views::iota(0, candidates)) {
        if (result.negative_area_fractions(k) <= max_negative_area) {
            result.chosen_mode = k;
            result.sign_consistent = true;
            break;
        }
    }
    Eigen::VectorXd const& x = vectors[result.chosen_mode];
    result.negative_area_fraction =
        result.negative_area_fractions(result.chosen_mode);
    result.rayleigh_quotient = x.dot(energy * x) / x.dot(norm * x);

    Eigen::VectorXd const delta = x.head(stacked_size);
    result.s = x.tail(face_count);
    result.residual_norm = std::sqrt(delta.dot(mass * delta));
    result.constraint_norm = (constraint * x).norm();
    return result;
}

// Continuation in μ: the smoothing needed for Eq. (7) to admit a
// sign-consistent (submersion-compatible) mode depends on the surface —
// a sphere needs μ ≈ 1e-4 (in total-area units) and would be over-smoothed
// by more, fertility needs ≈ 1e-2 or its lowest modes are localized bumps.
// Starting from `mu_start` and multiplying by 10 up to `mu_max` picks the
// smallest μ whose low spectrum contains a sign-consistent mode; if none
// does, the last result (lowest mode at mu_max) is returned.
inline InitialScaleResult initial_rescaling(
    PuncturedMesh const& punctured,
    FaceFieldOperators const& operators,
    double mu_start,
    double mu_max,
    int krylov_size,
    double tolerance,
    double max_negative_area = 0.02,
    double scalar_weight_floor = omega_floor)
{
    InitialScaleResult result;
    for (double mu = mu_start; mu <= mu_max * (1.0 + 1e-12); mu *= 10.0) {
        result = initial_rescaling_at(
            punctured,
            operators,
            mu,
            krylov_size,
            tolerance,
            max_negative_area,
            scalar_weight_floor);
        if (result.sign_consistent) {
            break;
        }
    }
    return result;
}

// §4.2.4, footnote 4: the largest phase change s ŵ⊥ · e over any edge.
inline double max_edge_phase(
    PuncturedMesh const& punctured,
    Eigen::VectorXd const& s)
{
    MeshTables const tables = mesh_tables(punctured.sub->mesh());
    Eigen::MatrixXd const perp = perpendicular_ambient(punctured);
    double rho = 0.0;
    for (int const interior :
        std::views::iota(0, static_cast<int>(tables.interior_edges.size()))) {
        int const edge = tables.interior_edges(interior);
        Eigen::RowVector3d const edge_vector =
            tables.vertex_positions.row(tables.edge_vertices(edge, 1))
            - tables.vertex_positions.row(tables.edge_vertices(edge, 0));
        for (int const side : std::views::iota(0, 2)) {
            int const face = tables.edge_faces(edge, side);
            rho = std::max(
                rho, std::abs(s(face) * perp.row(face).dot(edge_vector)));
        }
    }
    return rho;
}

struct RescaleResult {
    double rho = 0.0;
    double factor = 0.0;
};

// Scale s so the phase change across any edge is at most π·multiplier: θ
// is then never aliased on the mesh (multiplier ≤ 1) and the wrapped
// vertex differences used later are unambiguous.
inline RescaleResult global_rescale(
    PuncturedMesh const& punctured,
    Eigen::VectorXd& s,
    double multiplier)
{
    RescaleResult result;
    result.rho = max_edge_phase(punctured, s);
    if (result.rho <= 0.0) {
        throw std::runtime_error("global_rescale: s ŵ⊥ vanishes on every edge");
    }
    result.factor = multiplier * std::numbers::pi / result.rho;
    s *= result.factor;
    return result;
}
