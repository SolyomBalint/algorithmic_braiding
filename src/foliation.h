#pragma once

#include "face_field_operators.h"
#include "punctured_mesh.h"
#include "scale_field.h"

#include <directional/mass_matrices.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cmath>
#include <complex>
#include <functional>
#include <numbers>
#include <print>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

// Wrap to the principal branch [−π, π].
inline double wrap_angle(double x)
{
    constexpr double period = 2.0 * std::numbers::pi;
    return x - period * std::round(x / period);
}

struct Foliation {
    Eigen::VectorXd theta; // |V_sub|, values in [−π, π]
    Eigen::VectorXd s; // |F_sub|
};

// §4.2.4, Eq. (8): joint recovery of θ : V → S¹ and the rescaling s.
//
//   E(θ, s) = Σ_f Σ_{(α,β)∈f} (ω_{αβ}/2) ‖R(θ_β) − R(θ_α + s_f d_{f,αβ})‖²
//           + (μ h²/2) sᵀ Q_s s,
//   d_{f,αβ} = ŵ_f⊥ · (p_β − p_α),   R(φ) = (cos φ, sin φ),
//
// summed over the three directed corner pairs of each face; ω is the
// cotangent weight of the edge (α,β), h² the mean face area (μ is
// dimensionless, as in scale_field.h: the data term scales with d² ~ h²).
class FoliationSolver {
public:
    // Where the s-step linearizes the wrapped residual. `false`: about the
    // current s⁰ (targets s⁰d + wrap(Δθ − s⁰d), which can walk onto branches
    // with |s d| > π and lets s grow or flip sign). `true`: about s = 0
    // (targets wrap(Δθ) ∈ [−π, π), as in the reference code), which keeps
    // every edge phase within half a period, the assumption of the global
    // rescale.
    bool linearize_at_zero = false;

    // Prior on s in the s-step: (κ h²/2) ‖s − s⁰‖² with s⁰ the rescaled
    // stage-4 field and κ dimensionless like μ. Eq. (8) alone lets s drift
    // to zero wherever θ cannot follow the connection: the wrapped targets
    // there look like noise, the least-squares s becomes ≈ 0, the next
    // θ-step turns flat, and the leaves never come back (observed on
    // Thingi10k meshes: |s| < 0.1·max on 70–90 % of the area after ten
    // rounds, even from a strictly positive s⁰). κ = 0 is the paper's
    // energy.
    double prior_weight = 0.0;

    FoliationSolver(PuncturedMesh const& punctured, double mu)
        : punctured_(&punctured)
    {
        directional::TriMesh const& mesh = punctured.sub->mesh();
        MeshTables const tables = mesh_tables(mesh);
        mu_ = mu * tables.face_areas.mean();
        int const face_count = tables.faces.rows();
        Eigen::MatrixXd const perp = perpendicular_ambient(punctured);

        phase_per_unit_s_.resize(face_count, 3);
        weight_.resize(face_count, 3);
        for (int const face : std::views::iota(0, face_count)) {
            for (int const corner : std::views::iota(0, 3)) {
                int const alpha = tables.faces(face, corner);
                int const beta = tables.faces(face, (corner + 1) % 3);
                Eigen::RowVector3d const step
                    = tables.vertex_positions.row(beta)
                    - tables.vertex_positions.row(alpha);
                phase_per_unit_s_(face, corner) = perp.row(face).dot(step);
                weight_(face, corner) = edge_cotan_weight(
                    tables, edge_between(tables, face, alpha, beta));
            }
        }

        scalar_laplacian_ = face_scalar_laplacian(tables);
        vertex_mass_ = directional::lumped_voronoi_mass_matrix_2D<double>(mesh)
                           .diagonal();
    }

    // θ-step: with z_v = R(θ_v) ∈ R² and the unit constraint relaxed to
    // zᵀ (M_V ⊗ I₂) z = 1, the alignment term is ½ zᵀ L_d z with
    //   L[β,β] += ω I,  L[α,α] += ω I,  L[β,α] −= ω R(φ),  L[α,β] −= ω R(φ)ᵀ,
    //   φ = s_f d_{f,αβ}.
    // The minimizer is the smallest generalized eigenvector of (L_d, M_V⊗I₂),
    // found by inverse power iteration [Knöppel et al. 2015].
    // Returns the Rayleigh quotient zᵀ L_d z / zᵀ M z.
    double theta_step(Eigen::VectorXd const& s, int power_iterations,
        Eigen::VectorXd& theta) const
    {
        MeshTables const tables = mesh_tables(punctured_->sub->mesh());
        int const face_count = tables.faces.rows();
        int const vertex_count = tables.vertex_positions.rows();
        int const size = 2 * vertex_count;

        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(3 * face_count * 12);
        for (int const face : std::views::iota(0, face_count)) {
            for (int const corner : std::views::iota(0, 3)) {
                int const alpha = tables.faces(face, corner);
                int const beta = tables.faces(face, (corner + 1) % 3);
                double const omega = weight_(face, corner);
                double const phi = s(face) * phase_per_unit_s_(face, corner);
                double const c = std::cos(phi);
                double const sn = std::sin(phi);
                // R(φ) = [c −s; s c]
                entries.emplace_back(2 * alpha, 2 * alpha, omega);
                entries.emplace_back(2 * alpha + 1, 2 * alpha + 1, omega);
                entries.emplace_back(2 * beta, 2 * beta, omega);
                entries.emplace_back(2 * beta + 1, 2 * beta + 1, omega);
                // (β, α) block: −ω R(φ)
                entries.emplace_back(2 * beta, 2 * alpha, -omega * c);
                entries.emplace_back(2 * beta, 2 * alpha + 1, omega * sn);
                entries.emplace_back(2 * beta + 1, 2 * alpha, -omega * sn);
                entries.emplace_back(2 * beta + 1, 2 * alpha + 1, -omega * c);
                // (α, β) block: −ω R(φ)ᵀ
                entries.emplace_back(2 * alpha, 2 * beta, -omega * c);
                entries.emplace_back(2 * alpha, 2 * beta + 1, -omega * sn);
                entries.emplace_back(2 * alpha + 1, 2 * beta, omega * sn);
                entries.emplace_back(2 * alpha + 1, 2 * beta + 1, -omega * c);
            }
        }
        Eigen::SparseMatrix<double> laplacian(size, size);
        laplacian.setFromTriplets(entries.begin(), entries.end());

        Eigen::VectorXd mass(size);
        for (int const vertex : std::views::iota(0, vertex_count)) {
            mass(2 * vertex) = vertex_mass_(vertex);
            mass(2 * vertex + 1) = vertex_mass_(vertex);
        }

        // Tiny shift guards the exactly-integrable case (λ_min = 0).
        double const shift = 1e-8 * laplacian.diagonal().sum() / mass.sum();
        Eigen::SparseMatrix<double> shifted = laplacian;
        for (int const i : std::views::iota(0, size)) {
            shifted.coeffRef(i, i) += shift * mass(i);
        }
        shifted.makeCompressed();

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(shifted);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("theta_step: LDLT failed");
        }

        auto mass_norm = [&](Eigen::VectorXd const& z) {
            return std::sqrt(z.dot(mass.cwiseProduct(z)));
        };

        // Warm start from the previous θ when there is one, else random.
        Eigen::VectorXd z(size);
        if (theta.size() == vertex_count) {
            for (int const vertex : std::views::iota(0, vertex_count)) {
                z(2 * vertex) = std::cos(theta(vertex));
                z(2 * vertex + 1) = std::sin(theta(vertex));
            }
        } else {
            std::mt19937 rng(0);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (int const i : std::views::iota(0, size)) {
                z(i) = dist(rng);
            }
        }
        z /= mass_norm(z);
        for (int iteration = 0; iteration < power_iterations; ++iteration) {
            z = solver.solve(mass.cwiseProduct(z));
            z /= mass_norm(z);
        }
        double const rayleigh
            = z.dot(laplacian * z) / z.dot(mass.cwiseProduct(z));

        theta.resize(vertex_count);
        for (int const vertex : std::views::iota(0, vertex_count)) {
            theta(vertex) = std::atan2(z(2 * vertex + 1), z(2 * vertex));
        }
        return rayleigh;
    }

    // s-step: one Gauss–Newton step on the per-term residual
    //   r_{f,αβ}(s) = wrap(θ_β − θ_α − s_f d_{f,αβ}),
    // linearized about the current s⁰ (wrap is the identity on its branch):
    //   r ≈ t − s_f d,   t = s⁰_f d + wrap(θ_β − θ_α − s⁰_f d),
    // giving the SPD system ( diag_f Σ ω d² + μ Q_s ) s = b,  b_f = Σ ω d t.
    void s_step(Eigen::VectorXd const& theta, Eigen::VectorXd& s,
        Eigen::VectorXd const& prior = Eigen::VectorXd()) const
    {
        MeshTables const tables = mesh_tables(punctured_->sub->mesh());
        int const face_count = tables.faces.rows();
        double const kappa = prior_weight * tables.face_areas.mean();

        Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(face_count);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(face_count);
        for (int const face : std::views::iota(0, face_count)) {
            for (int const corner : std::views::iota(0, 3)) {
                int const alpha = tables.faces(face, corner);
                int const beta = tables.faces(face, (corner + 1) % 3);
                double const omega = weight_(face, corner);
                double const d = phase_per_unit_s_(face, corner);
                double const current = linearize_at_zero ? 0.0 : s(face) * d;
                double const target = current
                    + wrap_angle(theta(beta) - theta(alpha) - current);
                diagonal(face) += omega * d * d;
                rhs(face) += omega * d * target;
            }
        }

        Eigen::SparseMatrix<double> system = mu_ * scalar_laplacian_;
        bool const use_prior = kappa > 0.0 && prior.size() == face_count;
        for (int const face : std::views::iota(0, face_count)) {
            system.coeffRef(face, face) += diagonal(face);
            if (use_prior) {
                system.coeffRef(face, face) += kappa;
                rhs(face) += kappa * prior(face);
            }
        }
        system.makeCompressed();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(system);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("s_step: LDLT failed");
        }
        s = solver.solve(rhs);
    }

    [[nodiscard]] double energy(
        Eigen::VectorXd const& theta, Eigen::VectorXd const& s) const
    {
        MeshTables const tables = mesh_tables(punctured_->sub->mesh());
        int const face_count = tables.faces.rows();
        double alignment = 0.0;
        for (int const face : std::views::iota(0, face_count)) {
            for (int const corner : std::views::iota(0, 3)) {
                int const alpha = tables.faces(face, corner);
                int const beta = tables.faces(face, (corner + 1) % 3);
                double const mismatch = theta(beta) - theta(alpha)
                    - s(face) * phase_per_unit_s_(face, corner);
                // ‖R(a) − R(b)‖² = 2 − 2 cos(a − b)
                alignment += 0.5 * weight_(face, corner)
                    * (2.0 - 2.0 * std::cos(mismatch));
            }
        }
        return alignment + 0.5 * mu_ * s.dot(scalar_laplacian_ * s);
    }

    // Alternate θ- and s-steps, ending on a θ-step so θ matches the final s.
    [[nodiscard]] Foliation alternate(Eigen::VectorXd s, int alternations,
        int power_iterations,
        std::function<void(int, int, Foliation const&, double, double)> const&
            on_round
        = {}) const
    {
        Foliation foliation;
        foliation.s = std::move(s);
        Eigen::VectorXd const prior = foliation.s;
        Eigen::VectorXd previous_theta;
        for (int const round : std::views::iota(1, alternations + 1)) {
            double const rayleigh
                = theta_step(foliation.s, power_iterations, foliation.theta);
            // θ is defined up to a global phase: remove the best-fit
            // rotation before measuring the change.
            double theta_change = 0.0;
            if (previous_theta.size() == foliation.theta.size()) {
                std::complex<double> mean_rotation = 0.0;
                for (int const v : std::views::iota(
                         0, static_cast<int>(previous_theta.size()))) {
                    mean_rotation += std::polar(
                        1.0, foliation.theta(v) - previous_theta(v));
                }
                double const phase = std::arg(mean_rotation);
                for (int const v : std::views::iota(
                         0, static_cast<int>(previous_theta.size()))) {
                    theta_change += std::pow(wrap_angle(foliation.theta(v)
                                                 - previous_theta(v) - phase),
                        2);
                }
                theta_change = std::sqrt(theta_change);
            }
            previous_theta = foliation.theta;
            double const total = energy(foliation.theta, foliation.s);
            std::println("alternation {}: rayleigh {}  energy(8) {}  ‖Δθ‖ {}",
                round, rayleigh, total, theta_change);
            if (on_round) {
                on_round(round, alternations, foliation, total, rayleigh);
            }
            if (round == alternations) {
                break;
            }
            Eigen::VectorXd const before = foliation.s;
            s_step(foliation.theta, foliation.s, prior);
            std::println("alternation {}: ‖Δs‖ {}  energy(8) {}", round,
                (foliation.s - before).norm(),
                energy(foliation.theta, foliation.s));
        }
        return foliation;
    }

private:
    // The mesh edge joining two corners of `face`.
    static int edge_between(
        MeshTables const& tables, int face, int vertex_a, int vertex_b)
    {
        for (int const side : std::views::iota(0, 3)) {
            int const edge = tables.face_edges(face, side);
            int const start = tables.edge_vertices(edge, 0);
            int const end = tables.edge_vertices(edge, 1);
            if ((start == vertex_a && end == vertex_b)
                || (start == vertex_b && end == vertex_a)) {
                return edge;
            }
        }
        throw std::runtime_error("edge_between: corners not on an edge");
    }

    PuncturedMesh const* punctured_;
    double mu_; // μ h²
    Eigen::MatrixXd phase_per_unit_s_; // d_{f,αβ}, |F| × 3
    Eigen::MatrixXd weight_; // ω_{αβ}, |F| × 3
    Eigen::SparseMatrix<double> scalar_laplacian_; // Q_s
    Eigen::VectorXd vertex_mass_; // lumped M_V diagonal
};

// Per-face quality measures of a recovered foliation.
struct FoliationDiagnostics {
    Eigen::VectorXd alignment; // angle(∇θ, s ŵ⊥)/π ∈ [0, 1], 0 = aligned
    Eigen::VectorXi
        aliased; // 1 where the wrapped corner differences don't close
    int aliased_count = 0;
};

// Unwrap the three corner values of a face relative to corner 0.
// Returns false when the wrapped differences don't sum to zero, i.e. θ
// winds around inside the face (aliased).
inline bool unwrap_face(Eigen::VectorXd const& theta,
    Eigen::MatrixXi const& faces, int face, Eigen::Vector3d& unwrapped)
{
    double const t0 = theta(faces(face, 0));
    double const t1 = theta(faces(face, 1));
    double const t2 = theta(faces(face, 2));
    double const d01 = wrap_angle(t1 - t0);
    double const d12 = wrap_angle(t2 - t1);
    double const d20 = wrap_angle(t0 - t2);
    unwrapped << t0, t0 + d01, t0 + d01 + d12;
    return std::abs(d01 + d12 + d20) < 1e-9;
}

inline FoliationDiagnostics foliation_diagnostics(
    PuncturedMesh const& punctured, Foliation const& foliation)
{
    directional::TriMesh const& mesh = punctured.sub->mesh();
    int const face_count = mesh.F.rows();
    Eigen::MatrixXd const perp = perpendicular_ambient(punctured);

    FoliationDiagnostics diagnostics;
    diagnostics.alignment = Eigen::VectorXd::Zero(face_count);
    diagnostics.aliased = Eigen::VectorXi::Zero(face_count);

    for (int const face : std::views::iota(0, face_count)) {
        Eigen::Vector3d values;
        if (!unwrap_face(foliation.theta, mesh.F, face, values)) {
            diagnostics.aliased(face) = 1;
            ++diagnostics.aliased_count;
            diagnostics.alignment(face) = 1.0;
            continue;
        }
        Eigen::RowVector3d const p0 = mesh.V.row(mesh.F(face, 0));
        Eigen::RowVector3d const p1 = mesh.V.row(mesh.F(face, 1));
        Eigen::RowVector3d const p2 = mesh.V.row(mesh.F(face, 2));
        Eigen::RowVector3d const normal = mesh.faceNormals.row(face);
        double const area = mesh.faceAreas(face);
        // P1 gradient: ∇φ_j = n × e_j / (2A), e_j the edge opposite corner j.
        Eigen::RowVector3d const grad_phi1
            = normal.cross(p0 - p2) / (2.0 * area);
        Eigen::RowVector3d const grad_phi2
            = normal.cross(p1 - p0) / (2.0 * area);
        Eigen::RowVector3d const gradient = (values(1) - values(0)) * grad_phi1
            + (values(2) - values(0)) * grad_phi2;
        Eigen::RowVector3d const target = foliation.s(face) * perp.row(face);
        double const cross = gradient.cross(target).norm();
        double const dot = gradient.dot(target);
        diagnostics.alignment(face) = std::atan2(cross, dot) / std::numbers::pi;
    }
    return diagnostics;
}
