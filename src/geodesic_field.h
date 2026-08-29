#pragma once

#include "weaving_mesh.h"

#include <directional/CartesianField.h>

#include <Eigen/Dense>

#include <cmath>
#include <numbers>
#include <random>

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

        int const n_faces = weaving_mesh.mesh().F.rows();
        Eigen::MatrixXd int_field(n_faces, 2);

        std::uniform_real_distribution<double> dist(
            0.0, 2.0 * std::numbers::pi);

        for (int i = 0; i < n_faces; ++i) {
            double const theta = dist(rng_);
            int_field(i, 0) = std::cos(theta);
            int_field(i, 1) = std::sin(theta);
        }

        field_.set_intrinsic_field(int_field);
    }

    void perturb_random()
    {
        Eigen::MatrixXd int_field = field_.intField;
        std::uniform_real_distribution<double> dist(
            0.0, 2.0 * std::numbers::pi);

        for (int i = 0; i < int_field.rows(); ++i) {
            double const dtheta = dist(rng_);
            double const c = std::cos(dtheta);
            double const s = std::sin(dtheta);
            double const x = int_field(i, 0);
            double const y = int_field(i, 1);
            int_field(i, 0) = c * x - s * y;
            int_field(i, 1) = s * x + c * y;
        }

        field_.set_intrinsic_field(int_field);
    }

    directional::CartesianField const& field() const { return field_; }
    directional::CartesianField& field() { return field_; }

    // Per-face curl for display:
    // (∇×w)_f = (1/A_f) Σ_{neighbors n} (w_n - w_f) · e_{fn}
    // with e_{fn} oriented as in face f.  Ambient 3D, no transport.
    Eigen::VectorXd face_curl() const
    {
        directional::TriMesh const& mesh = weaving_mesh_->mesh();
        int const face_count = mesh.F.rows();
        Eigen::VectorXd curl_per_face(face_count);
        curl_per_face.setZero();

        for (int face = 0; face < face_count; ++face) {
            Eigen::RowVector3d const face_vector = field_.extField.row(face);
            double circulation = 0.0;
            for (int side = 0; side < 3; ++side) {
                int const neighbor = mesh.TT(face, side);
                if (neighbor < 0) {
                    continue;
                }
                int const edge = mesh.FE(face, side);
                Eigen::RowVector3d edge_vector =
                    mesh.V.row(mesh.EV(edge, 1)) - mesh.V.row(mesh.EV(edge, 0));
                if (mesh.FEs(face, side) < 0) {
                    edge_vector = -edge_vector;
                }
                circulation +=
                    (field_.extField.row(neighbor) - face_vector).dot(edge_vector);
            }
            curl_per_face(face) = circulation / mesh.faceAreas(face);
        }
        return curl_per_face;
    }

private:
    WeavingMesh const* weaving_mesh_;
    directional::CartesianField field_;
    std::mt19937 rng_;
};
