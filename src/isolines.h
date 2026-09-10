#pragma once

#include "foliation.h"

#include <directional/TriMesh.h>

#include <Eigen/Dense>

#include <cmath>
#include <numbers>
#include <ranges>
#include <vector>

struct IsolineCurves {
    Eigen::MatrixXd nodes; // 2·#segments × 3
    Eigen::MatrixXi edges; // #segments × 2
    Eigen::VectorXi level; // #segments, k ∈ [0, lines_per_period)
};

// Level sets of the periodic θ: the leaves of the foliation. Levels are
//   c_{k,m} = −π + 2πk/n + 2πm,  k = 0..n−1, m ∈ Z.
// Each face is unwrapped relative to its first corner so the ±π jump never
// produces a spurious crossing (Directional's isolines() interpolates raw
// values and would); then plain marching triangles per level.
inline IsolineCurves periodic_isolines(
    directional::TriMesh const& mesh,
    Eigen::VectorXd const& theta,
    int lines_per_period)
{
    constexpr double period = 2.0 * std::numbers::pi;
    double const spacing = period / lines_per_period;

    std::vector<Eigen::RowVector3d> nodes;
    std::vector<int> levels;

    int const face_count = mesh.F.rows();
    for (int const face : std::views::iota(0, face_count)) {
        Eigen::Vector3d values;
        if (!unwrap_face(theta, mesh.F, face, values)) {
            continue;
        }
        double const low = values.minCoeff();
        double const high = values.maxCoeff();
        // Level index range covering [low, high]:
        //   c = −π + j·spacing,  j = k + n m.
        int const first = static_cast<int>(
            std::ceil((low + std::numbers::pi) / spacing));
        int const last = static_cast<int>(
            std::floor((high + std::numbers::pi) / spacing));
        for (int const j : std::views::iota(first, last + 1)) {
            double const c = -std::numbers::pi + j * spacing;
            Eigen::RowVector3d crossing[2];
            int found = 0;
            for (int const corner : std::views::iota(0, 3)) {
                int const next = (corner + 1) % 3;
                double const a = values(corner);
                double const b = values(next);
                // Half-open test: a crossing exactly at a vertex is counted
                // once, on the edge where the vertex is the lower end.
                bool const crosses = (a <= c && c < b) || (b <= c && c < a);
                if (!crosses || found == 2) {
                    continue;
                }
                double const t = (c - a) / (b - a);
                crossing[found++] = (1.0 - t) * mesh.V.row(mesh.F(face, corner))
                    + t * mesh.V.row(mesh.F(face, next));
            }
            if (found != 2) {
                continue;
            }
            nodes.push_back(crossing[0]);
            nodes.push_back(crossing[1]);
            int const k = ((j % lines_per_period) + lines_per_period)
                % lines_per_period;
            levels.push_back(k);
        }
    }

    IsolineCurves curves;
    int const segment_count = static_cast<int>(levels.size());
    curves.nodes.resize(2 * segment_count, 3);
    curves.edges.resize(segment_count, 2);
    curves.level.resize(segment_count);
    for (int const i : std::views::iota(0, segment_count)) {
        curves.nodes.row(2 * i) = nodes[2 * i];
        curves.nodes.row(2 * i + 1) = nodes[2 * i + 1];
        curves.edges.row(i) << 2 * i, 2 * i + 1;
        curves.level(i) = levels[i];
    }
    return curves;
}
