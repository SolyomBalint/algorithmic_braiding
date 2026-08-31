#pragma once

#include "weaving_mesh.h"

#include <directional/CartesianField.h>
#include <directional/curl_matrices.h>

#include <Eigen/Sparse>

#include <array>
#include <cmath>
#include <complex>
#include <ranges>
#include <vector>

struct FaceFieldOperators {
    // C : |interior edges| × 2|F|
    // One scalar per interior edge: neighboring face vectors must agree
    // along that edge.  Cu = 0  ⇔  the stacked field u is curl-free.
    Eigen::SparseMatrix<double> curl_matrix;

    // M^v_F : 2|F| × 2|F|
    // Block-diagonal A_f I_2.  δᵀ M δ = Σ_f A_f ||δ_f||²,
    // the discrete surface integral of ||δ||².
    Eigen::SparseMatrix<double> mass_matrix;

    // Q = -L^v_F : 2|F| × 2|F|  (PSD)
    // uᵀ Q u = Σ_{edges} (1/ω) ||T_{right←left} u_left - u_right||².
    // Dual-graph Dirichlet energy of a face-based tangent field.
    Eigen::SparseMatrix<double> smoothness_matrix;
};

// Unit complex r = connection(edge)  →  SO(2) matrix T_{right←left}.
inline Eigen::Matrix2d rotation_from_connection(std::complex<double> const r)
{
    Eigen::Matrix2d transport;
    transport << r.real(), -r.imag(), r.imag(), r.real();
    return transport;
}

// The vertex of `face` that does not lie on the edge (vertex_a, vertex_b).
// That vertex is where the angle opposite the edge sits.
inline int opposite_vertex_in_face(
    MeshTables const& tables,
    int face,
    int vertex_a,
    int vertex_b)
{
    for (int const corner : std::views::iota(0, 3)) {
        int const vertex = tables.faces(face, corner);
        if (vertex != vertex_a && vertex != vertex_b) {
            return vertex;
        }
    }
    return -1;
}

// cot of the angle at `apex` in triangle apex–vertex_a–vertex_b.
inline double cotan_at_apex(
    MeshTables const& tables,
    int apex,
    int vertex_a,
    int vertex_b)
{
    Eigen::RowVector3d const to_a =
        tables.vertex_positions.row(vertex_a)
        - tables.vertex_positions.row(apex);
    Eigen::RowVector3d const to_b =
        tables.vertex_positions.row(vertex_b)
        - tables.vertex_positions.row(apex);
    double const area_twice = to_a.cross(to_b).norm();
    return to_a.dot(to_b) / std::max(area_twice, 1e-16);
}

inline FaceFieldOperators assemble_face_field_operators(
    WeavingMesh const& weaving_mesh)
{
    directional::TriMesh const& mesh = weaving_mesh.mesh();
    MeshTables const tables = mesh_tables(mesh);
    directional::PCFaceTangentBundle const& tangent_bundle =
        weaving_mesh.tangent_bundle();

    int const face_count = tables.faces.rows();
    int const interior_edge_count = tables.interior_edges.size();

    FaceFieldOperators operators;

    operators.curl_matrix =
        directional::curl_matrix_2D<double>(mesh, /*isIntrinsic=*/true);

    std::vector<Eigen::Triplet<double>> mass_entries;
    mass_entries.reserve(2 * face_count);
    for (int const face : std::views::iota(0, face_count)) {
        double const area = tables.face_areas(face);
        mass_entries.emplace_back(2 * face, 2 * face, area);
        mass_entries.emplace_back(2 * face + 1, 2 * face + 1, area);
    }
    operators.mass_matrix.resize(2 * face_count, 2 * face_count);
    operators.mass_matrix.setFromTriplets(
        mass_entries.begin(), mass_entries.end());

    std::vector<Eigen::Triplet<double>> smoothness_entries;
    constexpr double omega_floor = 1e-8;

    for (int const interior : std::views::iota(0, interior_edge_count)) {
        int const edge = tables.interior_edges(interior);
        int const left_face = tables.edge_faces(edge, 0);
        int const right_face = tables.edge_faces(edge, 1);
        int const edge_start = tables.edge_vertices(edge, 0);
        int const edge_end = tables.edge_vertices(edge, 1);

        int const left_apex =
            opposite_vertex_in_face(tables, left_face, edge_start, edge_end);
        int const right_apex =
            opposite_vertex_in_face(tables, right_face, edge_start, edge_end);

        // ω = (cot α + cot β)/2, α,β opposite the shared edge.
        // Reciprocal 1/ω is the dual-graph weight on this face-adjacency.
        double omega =
            0.5
            * (cotan_at_apex(tables, left_apex, edge_start, edge_end)
               + cotan_at_apex(tables, right_apex, edge_start, edge_end));
        omega = std::max(omega, omega_floor);
        double const dual_weight = 1.0 / omega;

        // Unfold left onto right, then compare intrinsic 2-vectors.
        Eigen::Matrix2d const transport =
            rotation_from_connection(tangent_bundle.connection(edge));

        // local stamp [T  -I] applied to (u_left; u_right).
        Eigen::Matrix<double, 2, 4> stamp;
        stamp.block<2, 2>(0, 0) = transport;
        stamp.block<2, 2>(0, 2) = -Eigen::Matrix2d::Identity();
        Eigen::Matrix4d const local_energy =
            dual_weight * stamp.transpose() * stamp;

        std::array const stacked_cols{
            2 * left_face,
            2 * left_face + 1,
            2 * right_face,
            2 * right_face + 1};
        for (int const row : std::views::iota(0, 4)) {
            for (int const col : std::views::iota(0, 4)) {
                if (local_energy(row, col) != 0.0) {
                    smoothness_entries.emplace_back(
                        stacked_cols[row],
                        stacked_cols[col],
                        local_energy(row, col));
                }
            }
        }
    }

    operators.smoothness_matrix.resize(2 * face_count, 2 * face_count);
    operators.smoothness_matrix.setFromTriplets(
        smoothness_entries.begin(), smoothness_entries.end());

    return operators;
}

// intField (|F|×2) → column u ∈ R^{2|F|}, face-major (x_f, y_f).
// Same layout as C, M^v_F, Q.
inline Eigen::VectorXd flatten_intrinsic(
    directional::CartesianField const& field)
{
    return field.flatten(/*isIntrinsic=*/true);
}

inline Eigen::MatrixXd unflatten_intrinsic(
    Eigen::VectorXd const& stacked,
    int face_count)
{
    using RowMajorF2 =
        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
    return Eigen::Map<RowMajorF2 const>(stacked.data(), face_count, 2);
}
