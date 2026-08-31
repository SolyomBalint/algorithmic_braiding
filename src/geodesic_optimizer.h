#pragma once

#include "face_field_operators.h"
#include "geodesic_field.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include <algorithm>
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

    int run(
        double lambda,
        double epsilon,
        int max_iterations,
        bool enforce_curl)
    {
        int iterations = 0;
        for (int const iteration :
            std::views::iota(1, max_iterations + 1)) {
            iterations = iteration;
            double const field_change =
                take_step(lambda, enforce_curl, iteration);
            if (field_change < epsilon) {
                break;
            }
        }
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

private:
    // App. B typesets (M + λ L) δ = -λ L (u+δ). L is NSD and δ is unknown:
    // both wrong. Stationarity of Eq. (5) is this KKT with Q = -L (PSD):
    //   [ M+λQ   Cᵀ ] [δ̃]   [ -λ Q u ]
    //   [ C       0 ] [μ] = [ -C u   ]
    // SIGGRAPH uses Eigen::SPQR (SuiteSparse). Eigen::SparseQR is the
    // same rank-revealing QR; δ̃ is unique either way.
    void factor_kkt(double lambda)
    {
        Eigen::SparseMatrix<double> const& mass =
            field_operators_.mass_matrix;
        Eigen::SparseMatrix<double> const& smoothness =
            field_operators_.smoothness_matrix;
        Eigen::SparseMatrix<double> const& curl =
            field_operators_.curl_matrix;

        Eigen::SparseMatrix<double> const hessian =
            mass + lambda * smoothness;

        int const stacked_size = hessian.rows();
        int const curl_rows = curl.rows();
        int const kkt_size = stacked_size + curl_rows;

        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(hessian.nonZeros() + 2 * curl.nonZeros());

        for (int col = 0; col < hessian.outerSize(); ++col) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, col);
                 it;
                 ++it) {
                entries.emplace_back(it.row(), it.col(), it.value());
            }
        }
        for (int col = 0; col < curl.outerSize(); ++col) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(curl, col);
                 it;
                 ++it) {
                entries.emplace_back(
                    it.row() + stacked_size, it.col(), it.value());
                entries.emplace_back(
                    it.col(), it.row() + stacked_size, it.value());
            }
        }

        Eigen::SparseMatrix<double> kkt(kkt_size, kkt_size);
        kkt.setFromTriplets(entries.begin(), entries.end());
        kkt.makeCompressed();

        solver_.compute(kkt);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("KKT SparseQR factorization failed");
        }

        factored_lambda_ = lambda;
    }

    void factor_unconstrained(double lambda)
    {
        Eigen::SparseMatrix<double> const hessian =
            field_operators_.mass_matrix
            + lambda * field_operators_.smoothness_matrix;
        unconstrained_solver_.compute(hessian);
        if (unconstrained_solver_.info() != Eigen::Success) {
            throw std::runtime_error(
                "unconstrained LDLT factorization failed");
        }
        unconstrained_factored_lambda_ = lambda;
    }

    [[nodiscard]] double take_step(
        double lambda,
        bool enforce_curl,
        int iteration)
    {
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
            Eigen::VectorXd const solution = solver_.solve(rhs);
            corrected = design_field + solution.head(stacked_size);
        } else {
            if (unconstrained_factored_lambda_ != lambda) {
                factor_unconstrained(lambda);
            }
            Eigen::VectorXd const delta_tilde = unconstrained_solver_.solve(
                -lambda * smoothness * design_field);
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
        print_energy(
            lambda,
            enforce_curl,
            iteration,
            field_change,
            design_field);
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
        double field_change,
        Eigen::VectorXd const& design_field) const
    {
        Eigen::VectorXd const corrected = design_field + residual_;
        double const mass_term = 0.5
            * residual_.dot(field_operators_.mass_matrix * residual_);
        double const smoothness_term = 0.5 * lambda
            * corrected.dot(
                field_operators_.smoothness_matrix * corrected);
        double const curl_norm =
            (field_operators_.curl_matrix * corrected).norm();
        std::println(
            "λ={}  {}  it={}  ‖Δu‖_M={}  energy {} + {} = {}  ||C(u+δ)|| {}",
            lambda,
            enforce_curl ? "C" : "no-C",
            iteration,
            field_change,
            mass_term,
            smoothness_term,
            mass_term + smoothness_term,
            curl_norm);
    }

    GeodesicField& geodesic_field_;
    FaceFieldOperators const& field_operators_;
    Eigen::VectorXd residual_;
    Eigen::SparseQR<
        Eigen::SparseMatrix<double>,
        Eigen::COLAMDOrdering<int>>
        solver_;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>
        unconstrained_solver_;
    std::optional<double> factored_lambda_;
    std::optional<double> unconstrained_factored_lambda_;
};
