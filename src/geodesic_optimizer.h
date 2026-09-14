#pragma once

#include "face_field_operators.h"
#include "geodesic_field.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>
#include <vector>

class GeodesicOptimizer {
public:
    GeodesicOptimizer(
        GeodesicField& geodesic_field,
        FaceFieldOperators const& field_operators)
        : geodesic_field_(geodesic_field)
        , field_operators_(field_operators)
        , residual_(Eigen::VectorXd::Zero(
              field_operators.mass_matrix.rows()))
        // M is diag(A_f, A_f, ...): each face area appears twice.
        , total_area_(field_operators.mass_matrix.diagonal().sum() / 2.0)
    {
    }

    // One Algorithm 1 iteration: δ-step (KKT) then u-step (unit circle).
    // Returns ‖u^i − u^{i-1}‖_M.
    [[nodiscard]] double step(double lambda)
    {
        return take_step(lambda, true, 0);
    }

    [[nodiscard]] double step_unconstrained(double lambda)
    {
        return take_step(lambda, false, 0);
    }

    // `epsilon` is an average per-face change: the stop test compares
    // ‖Δu‖_M against ε·√(total area), so it is independent of mesh scale.
    int run(
        double lambda,
        double epsilon,
        int max_iterations,
        bool enforce_curl)
    {
        using clock = std::chrono::steady_clock;
        auto const level_start = clock::now();
        solve_seconds_ = 0.0;
        factor_seconds_ = 0.0;

        double const threshold = epsilon * std::sqrt(total_area_);
        int iterations = 0;
        for (int const iteration :
            std::views::iota(1, max_iterations + 1)) {
            iterations = iteration;
            double const field_change =
                take_step(lambda, enforce_curl, iteration);
            if (field_change < threshold) {
                break;
            }
        }

        double const level_seconds =
            std::chrono::duration<double>(clock::now() - level_start).count();
        std::println(
            "λ={}  {}  level done: {} iterations, factor {:.3f}s, "
            "solves {:.3f}s, total {:.3f}s",
            lambda,
            enforce_curl ? "C" : "no-C",
            iterations,
            factor_seconds_,
            solve_seconds_,
            level_seconds);
        return iterations;
    }

    void optimize(
        double lambda_start,
        double lambda_end,
        double shrink,
        double epsilon,
        int max_iterations)
    {
        if (shrink <= 1.0) {
            throw std::runtime_error("shrink must be > 1");
        }
        if (lambda_start < lambda_end) {
            throw std::runtime_error(
                "lambda_start must be >= lambda_end");
        }

        run(lambda_start, epsilon, max_iterations, false);

        double lambda = lambda_start;
        while (true) {
            run(lambda, epsilon, max_iterations, true);
            if (lambda <= lambda_end) {
                break;
            }
            lambda = std::max(lambda / shrink, lambda_end);
        }
    }

    [[nodiscard]] Eigen::VectorXd const& residual() const
    {
        return residual_;
    }

    struct EnergyBreakdown {
        double mass_term = 0.0;
        double smoothness_term = 0.0;
        double total = 0.0;
        double curl_norm = 0.0;
    };

    [[nodiscard]] EnergyBreakdown energy(double lambda) const
    {
        Eigen::VectorXd const design_field =
            flatten_intrinsic(geodesic_field_.field());
        Eigen::VectorXd const corrected = design_field + residual_;
        EnergyBreakdown out;
        out.mass_term = 0.5
            * residual_.dot(field_operators_.mass_matrix * residual_);
        out.smoothness_term = 0.5 * lambda
            * corrected.dot(
                field_operators_.smoothness_matrix * corrected);
        out.total = out.mass_term + out.smoothness_term;
        out.curl_norm =
            (field_operators_.curl_matrix * corrected).norm();
        return out;
    }

    [[nodiscard]] double mass_norm_threshold(double epsilon) const
    {
        return epsilon * std::sqrt(total_area_);
    }

private:
    // App. B typesets (M + λ L) δ = -λ L (u+δ). L is NSD and δ is unknown:
    // both wrong. Stationarity of Eq. (5) is this KKT with Q = -L (PSD):
    //   [ M+λQ   Cᵀ  ] [δ̃]   [ -λ Q u ]
    //   [ C     -εI  ] [μ] = [ -C u   ]
    // On a closed surface rank(C) = |E_int| − 1, so with a zero (2,2)
    // block the KKT matrix is singular (δ̃ unique, μ not). The paper
    // sidesteps that with a rank-revealing sparse QR (SuiteSparse SPQR);
    // Eigen's SparseQR is far slower. Instead we stamp −εI: the matrix
    // becomes quasi-definite (SPD block, negative-definite block), which
    // has a stable LDLᵀ factorization under any symmetric ordering, so
    // SimplicialLDLT applies directly. The price is C(u+δ̃) = εμ instead
    // of 0, negligible for ε ≪ scale of M+λQ.
    void factor_kkt(double lambda)
    {
        using clock = std::chrono::steady_clock;
        auto const start = clock::now();

        Eigen::SparseMatrix<double> const& mass =
            field_operators_.mass_matrix;
        Eigen::SparseMatrix<double> const& smoothness =
            field_operators_.smoothness_matrix;
        Eigen::SparseMatrix<double> const& curl =
            field_operators_.curl_matrix;

        Eigen::SparseMatrix<double> const hessian =
            mass + lambda * smoothness;

        // ε is tied to the mass scale (mean face area), not to M+λQ: the
        // smoothness weights 1/ω can be orders of magnitude larger than M
        // on meshes with near-right angles, and an ε scaled by them lets
        // the constraint drift by O(ε‖μ‖).
        constexpr double relative_regularization = 1e-10;
        double const regularization = relative_regularization
            * mass.diagonal().sum() / static_cast<double>(hessian.rows());

        Eigen::SparseMatrix<double> const kkt =
            assemble_quasi_definite_kkt(hessian, curl, regularization);

        solver_.compute(kkt);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("KKT LDLT factorization failed");
        }

        factored_lambda_ = lambda;
        factor_seconds_ +=
            std::chrono::duration<double>(clock::now() - start).count();
    }

    void factor_unconstrained(double lambda)
    {
        using clock = std::chrono::steady_clock;
        auto const start = clock::now();

        Eigen::SparseMatrix<double> const hessian =
            field_operators_.mass_matrix
            + lambda * field_operators_.smoothness_matrix;
        unconstrained_solver_.compute(hessian);
        if (unconstrained_solver_.info() != Eigen::Success) {
            throw std::runtime_error(
                "unconstrained LDLT factorization failed");
        }
        unconstrained_factored_lambda_ = lambda;
        factor_seconds_ +=
            std::chrono::duration<double>(clock::now() - start).count();
    }

    [[nodiscard]] double take_step(
        double lambda,
        bool enforce_curl,
        int iteration)
    {
        using clock = std::chrono::steady_clock;

        Eigen::VectorXd design_field =
            flatten_intrinsic(geodesic_field_.field());
        Eigen::VectorXd const previous = design_field;

        Eigen::SparseMatrix<double> const& smoothness =
            field_operators_.smoothness_matrix;
        Eigen::VectorXd corrected;
        if (enforce_curl) {
            if (factored_lambda_ != lambda) {
                factor_kkt(lambda);
            }
            Eigen::SparseMatrix<double> const& curl =
                field_operators_.curl_matrix;
            int const stacked_size = design_field.size();
            int const curl_rows = curl.rows();
            Eigen::VectorXd rhs(stacked_size + curl_rows);
            rhs.head(stacked_size) = -lambda * smoothness * design_field;
            rhs.tail(curl_rows) = -curl * design_field;

            auto const solve_start = clock::now();
            Eigen::VectorXd const solution = solver_.solve(rhs);
            solve_seconds_ += std::chrono::duration<double>(
                clock::now() - solve_start)
                                  .count();
            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error("KKT LDLT solve failed");
            }
            corrected = design_field + solution.head(stacked_size);
        } else {
            if (unconstrained_factored_lambda_ != lambda) {
                factor_unconstrained(lambda);
            }
            auto const solve_start = clock::now();
            Eigen::VectorXd const delta_tilde = unconstrained_solver_.solve(
                -lambda * smoothness * design_field);
            solve_seconds_ += std::chrono::duration<double>(
                clock::now() - solve_start)
                                  .count();
            if (unconstrained_solver_.info() != Eigen::Success) {
                throw std::runtime_error("unconstrained LDLT solve failed");
            }
            corrected = design_field + delta_tilde;
        }

        project_to_unit(design_field, corrected);
        residual_ = corrected - design_field;
        geodesic_field_.field().set_intrinsic_field(unflatten_intrinsic(
            design_field, design_field.size() / 2));

        Eigen::VectorXd const delta_u = design_field - previous;
        double const field_change = std::sqrt(
            delta_u.dot(field_operators_.mass_matrix * delta_u));
        print_energy(lambda, enforce_curl, iteration, field_change);
        return field_change;
    }

    static void project_to_unit(
        Eigen::VectorXd& design_field,
        Eigen::VectorXd const& corrected)
    {
        int const face_count = design_field.size() / 2;
        constexpr double length_floor = 1e-12;
        for (int const face : std::views::iota(0, face_count)) {
            Eigen::Vector2d const corrected_face(
                corrected(2 * face), corrected(2 * face + 1));
            double const length = corrected_face.norm();
            if (length < length_floor) {
                continue;
            }
            design_field(2 * face) = corrected_face(0) / length;
            design_field(2 * face + 1) = corrected_face(1) / length;
        }
    }

    void print_energy(
        double lambda,
        bool enforce_curl,
        int iteration,
        double field_change) const
    {
        EnergyBreakdown const terms = energy(lambda);
        std::println(
            "λ={}  {}  it={}  ‖Δu‖_M={}  energy {} + {} = {}  ||C(u+δ)|| {}",
            lambda,
            enforce_curl ? "C" : "no-C",
            iteration,
            field_change,
            terms.mass_term,
            terms.smoothness_term,
            terms.total,
            terms.curl_norm);
    }

    GeodesicField& geodesic_field_;
    FaceFieldOperators const& field_operators_;
    Eigen::VectorXd residual_;
    double total_area_;
    double factor_seconds_ = 0.0;
    double solve_seconds_ = 0.0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>
        unconstrained_solver_;
    std::optional<double> factored_lambda_;
    std::optional<double> unconstrained_factored_lambda_;
};
