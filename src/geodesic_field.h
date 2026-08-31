#pragma once

#include "weaving_mesh.h"

#include <directional/CartesianField.h>

#include <Eigen/Dense>

#include <complex>
#include <numbers>
#include <random>
#include <ranges>

class GeodesicField {
public:
    explicit GeodesicField(WeavingMesh const& weaving_mesh)
        : weaving_mesh_(&weaving_mesh)
        , rng_(std::random_device{}())
    {
        field_.init(
            weaving_mesh.tangent_bundle(),
            directional::fieldTypeEnum::RAW_FIELD,
            1);

        MeshTables const tables = mesh_tables(weaving_mesh.mesh());
        int const n_faces = tables.faces.rows();
        Eigen::MatrixXd int_field(n_faces, 2);

        std::uniform_real_distribution<double> dist(
            0.0, 2.0 * std::numbers::pi);

        for (int const face : std::views::iota(0, n_faces)) {
            std::complex<double> const z = std::polar(1.0, dist(rng_));
            int_field(face, 0) = z.real();
            int_field(face, 1) = z.imag();
        }

        field_.set_intrinsic_field(int_field);
    }

    void perturb_random()
    {
        Eigen::MatrixXd int_field = field_.intField;
        std::uniform_real_distribution<double> dist(
            0.0, 2.0 * std::numbers::pi);

        for (int const face : std::views::iota(0, int_field.rows())) {
            std::complex<double> const rotation =
                std::polar(1.0, dist(rng_));
            std::complex<double> const vector(
                int_field(face, 0), int_field(face, 1));
            std::complex<double> const rotated = rotation * vector;
            int_field(face, 0) = rotated.real();
            int_field(face, 1) = rotated.imag();
        }

        field_.set_intrinsic_field(int_field);
    }

    [[nodiscard]] directional::CartesianField const& field() const
    {
        return field_;
    }
    [[nodiscard]] directional::CartesianField& field() { return field_; }

    // Per-face curl for display:
    // (∇×w)_f = (1/A_f) Σ_{neighbors n} (w_n - w_f) · e_{fn}
    // with e_{fn} oriented as in face f.  Ambient 3D, no transport.
    [[nodiscard]] Eigen::VectorXd face_curl() const
    {
        MeshTables const tables = mesh_tables(weaving_mesh_->mesh());
        int const face_count = tables.faces.rows();
        Eigen::VectorXd curl_per_face(face_count);
        curl_per_face.setZero();

        for (int const face : std::views::iota(0, face_count)) {
            Eigen::RowVector3d const face_vector = field_.extField.row(face);
            double circulation = 0.0;
            for (int const side : std::views::iota(0, 3)) {
                int const neighbor = tables.face_neighbors(face, side);
                if (neighbor < 0) {
                    continue;
                }
                int const edge = tables.face_edges(face, side);
                Eigen::RowVector3d edge_vector =
                    tables.vertex_positions.row(tables.edge_vertices(edge, 1))
                    - tables.vertex_positions.row(
                        tables.edge_vertices(edge, 0));
                if (tables.face_edge_signs(face, side) < 0) {
                    edge_vector = -edge_vector;
                }
                circulation += (field_.extField.row(neighbor) - face_vector)
                                   .dot(edge_vector);
            }
            curl_per_face(face) = circulation / tables.face_areas(face);
        }
        return curl_per_face;
    }

private:
    WeavingMesh const* weaving_mesh_;
    directional::CartesianField field_;
    std::mt19937 rng_;
};
