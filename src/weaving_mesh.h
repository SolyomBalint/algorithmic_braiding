#pragma once

#include <directional/TriMesh.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/readOBJ.h>

#include <stdexcept>
#include <string>

class WeavingMesh {
public:
    explicit WeavingMesh(std::string const& path)
    {
        if (!directional::readOBJ(path, mesh_)) {
            throw std::runtime_error(path);
        }
        tangent_bundle_.init(mesh_);
    }

    // Build from explicit geometry (used for the punctured sub-mesh).
    WeavingMesh(Eigen::MatrixXd const& vertices, Eigen::MatrixXi const& faces)
    {
        mesh_.set_mesh(vertices, faces);
        tangent_bundle_.init(mesh_);
    }

    WeavingMesh(WeavingMesh const&) = delete;
    WeavingMesh& operator=(WeavingMesh const&) = delete;
    WeavingMesh(WeavingMesh&&) = delete;
    WeavingMesh& operator=(WeavingMesh&&) = delete;

    [[nodiscard]] directional::TriMesh const& mesh() const { return mesh_; }
    [[nodiscard]] directional::TriMesh& mesh() { return mesh_; }

    [[nodiscard]] directional::PCFaceTangentBundle const& tangent_bundle()
        const
    {
        return tangent_bundle_;
    }

private:
    directional::TriMesh mesh_;
    directional::PCFaceTangentBundle tangent_bundle_;
};

// Directional stores adjacency in terse tables (V, F, EV, …). Bind them
// once so call sites can read the geometry instead of the abbreviations.
struct MeshTables {
    Eigen::MatrixXd const& vertex_positions; // V
    Eigen::MatrixXi const& faces; // F
    Eigen::MatrixXi const& edge_vertices; // EV  |E|×2  start, end
    Eigen::MatrixXi const& edge_faces; // EF  |E|×2  left, right
    Eigen::MatrixXi const& face_edges; // FE  |F|×3
    Eigen::MatrixXd const& face_edge_signs; // FEs  +1 if FE matches EV
    Eigen::MatrixXi const& face_neighbors; // TT  neighbor across side
    Eigen::VectorXi const& interior_edges; // innerEdges
    Eigen::VectorXd const& face_areas; // faceAreas
};

inline MeshTables mesh_tables(directional::TriMesh const& mesh)
{
    return {
        .vertex_positions = mesh.V,
        .faces = mesh.F,
        .edge_vertices = mesh.EV,
        .edge_faces = mesh.EF,
        .face_edges = mesh.FE,
        .face_edge_signs = mesh.FEs,
        .face_neighbors = mesh.TT,
        .interior_edges = mesh.innerEdges,
        .face_areas = mesh.faceAreas,
    };
}
