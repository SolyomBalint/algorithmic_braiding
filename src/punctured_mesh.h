#pragma once

#include "geodesic_field.h"
#include "weaving_mesh.h"

#include <directional/CartesianField.h>

#include <Eigen/Dense>

#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

// §4.2.2: the surface with every singular vertex of ŵ and its incident faces
// removed. θ is only expected to be smooth on this punctured surface.
struct PuncturedMesh {
    std::unique_ptr<WeavingMesh> sub; // punctured surface
    directional::CartesianField field; // ŵ restricted to `sub` (RAW, N=1)
    Eigen::VectorXi face_to_original; // |F_sub|
    Eigen::VectorXi vertex_to_original; // |V_sub|
    Eigen::VectorXi original_to_vertex; // |V_orig|, −1 if deleted
    Eigen::VectorXi deleted_faces; // original face indices
};

// Face across side `side` of `face`, or −1 on the boundary. Directional's
// TT is left uninitialized on boundary sides, so go through FE/EF instead.
inline int face_across(directional::TriMesh const& mesh, int face, int side)
{
    int const edge = mesh.FE(face, side);
    return mesh.EF(edge, 0) == face ? mesh.EF(edge, 1) : mesh.EF(edge, 0);
}

// Number of connected components of the face-adjacency graph.
inline int face_component_count(directional::TriMesh const& mesh)
{
    int const face_count = mesh.F.rows();
    std::vector<int> label(face_count, -1);
    int components = 0;
    std::vector<int> stack;
    for (int const seed : std::views::iota(0, face_count)) {
        if (label[seed] >= 0) {
            continue;
        }
        label[seed] = components;
        stack.push_back(seed);
        while (!stack.empty()) {
            int const face = stack.back();
            stack.pop_back();
            for (int const side : std::views::iota(0, 3)) {
                int const neighbor = face_across(mesh, face, side);
                if (neighbor >= 0 && label[neighbor] < 0) {
                    label[neighbor] = components;
                    stack.push_back(neighbor);
                }
            }
        }
        ++components;
    }
    return components;
}

// `rings` = 1 removes the faces incident to each singular vertex (§4.2.2);
// larger values also remove the faces incident to the previous ring.
inline PuncturedMesh puncture(
    WeavingMesh const& original,
    GeodesicField const& geodesic_field,
    int rings = 1)
{
    MeshTables const tables = mesh_tables(original.mesh());
    directional::CartesianField const& field = geodesic_field.field();
    int const vertex_count = tables.vertex_positions.rows();
    int const face_count = tables.faces.rows();

    // singLocalCycles are vertex indices for a face-based tangent bundle
    // (its cycleSources are the mesh vertices).
    std::vector<char> singular(vertex_count, 0);
    for (int const i :
        std::views::iota(0, static_cast<int>(field.singLocalCycles.size()))) {
        if (field.singIndices(i) != 0) {
            singular[field.singLocalCycles(i)] = 1;
        }
    }

    std::vector<char> keep_face(face_count, 1);
    std::vector<char> removed_vertex = singular;
    for (int ring = 0; ring < rings; ++ring) {
        std::vector<char> next_removed = removed_vertex;
        for (int const face : std::views::iota(0, face_count)) {
            bool touches = false;
            for (int const corner : std::views::iota(0, 3)) {
                touches = touches || removed_vertex[tables.faces(face, corner)];
            }
            if (!touches) {
                continue;
            }
            keep_face[face] = 0;
            for (int const corner : std::views::iota(0, 3)) {
                next_removed[tables.faces(face, corner)] = 1;
            }
        }
        removed_vertex = next_removed;
    }

    PuncturedMesh result;
    result.original_to_vertex = Eigen::VectorXi::Constant(vertex_count, -1);
    int kept_faces = 0;
    int kept_vertices = 0;
    for (int const face : std::views::iota(0, face_count)) {
        if (!keep_face[face]) {
            continue;
        }
        ++kept_faces;
        for (int const corner : std::views::iota(0, 3)) {
            int const vertex = tables.faces(face, corner);
            if (result.original_to_vertex(vertex) < 0) {
                result.original_to_vertex(vertex) = kept_vertices++;
            }
        }
    }
    if (kept_faces == 0) {
        throw std::runtime_error("puncture: no faces left");
    }

    Eigen::MatrixXd vertices(kept_vertices, 3);
    Eigen::MatrixXi faces(kept_faces, 3);
    Eigen::MatrixXd extrinsic(kept_faces, 3);
    result.vertex_to_original.resize(kept_vertices);
    result.face_to_original.resize(kept_faces);
    result.deleted_faces.resize(face_count - kept_faces);

    for (int const vertex : std::views::iota(0, vertex_count)) {
        int const sub_vertex = result.original_to_vertex(vertex);
        if (sub_vertex >= 0) {
            result.vertex_to_original(sub_vertex) = vertex;
            vertices.row(sub_vertex) = tables.vertex_positions.row(vertex);
        }
    }
    int sub_face = 0;
    int deleted = 0;
    for (int const face : std::views::iota(0, face_count)) {
        if (!keep_face[face]) {
            result.deleted_faces(deleted++) = face;
            continue;
        }
        result.face_to_original(sub_face) = face;
        for (int const corner : std::views::iota(0, 3)) {
            faces(sub_face, corner) =
                result.original_to_vertex(tables.faces(face, corner));
        }
        extrinsic.row(sub_face) = field.extField.row(face);
        ++sub_face;
    }

    result.sub = std::make_unique<WeavingMesh>(vertices, faces);
    result.field.init(
        result.sub->tangent_bundle(), directional::fieldTypeEnum::RAW_FIELD, 1);
    // The face planes are unchanged, so projecting the ambient vectors onto
    // the new per-face frames loses nothing.
    result.field.set_extrinsic_field(extrinsic);

    int const components = face_component_count(result.sub->mesh());
    if (components != 1) {
        throw std::runtime_error(
            "puncture: punctured surface has " + std::to_string(components)
            + " connected components; only one is supported");
    }
    return result;
}
