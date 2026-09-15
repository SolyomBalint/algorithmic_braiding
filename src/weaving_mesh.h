#pragma once

#include <directional/TriMesh.h>
#include <directional/PCFaceTangentBundle.h>

#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline std::string mesh_extension(std::string const& path)
{
    auto const pos = path.find_last_of('.');
    if (pos == std::string::npos) {
        return {};
    }
    std::string ext = path.substr(pos);
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

inline void validate_triangle_mesh(
    Eigen::MatrixXd const& vertices,
    Eigen::MatrixXi const& faces,
    std::string const& path)
{
    if (vertices.rows() < 3 || vertices.cols() != 3) {
        throw std::runtime_error(path + ": need at least 3 vertices");
    }
    if (faces.rows() < 1 || faces.cols() != 3) {
        throw std::runtime_error(path + ": need at least one triangle");
    }
    if (faces.minCoeff() < 0 || faces.maxCoeff() >= vertices.rows()) {
        throw std::runtime_error(path + ": face index out of range");
    }

    std::vector<char> used(static_cast<std::size_t>(vertices.rows()), 0);
    std::map<std::pair<int, int>, int> edge_face_count;
    constexpr double area2_floor = 1e-30;
    for (int face = 0; face < faces.rows(); ++face) {
        int const a = faces(face, 0);
        int const b = faces(face, 1);
        int const c = faces(face, 2);
        if (a == b || b == c || a == c) {
            throw std::runtime_error(path + ": degenerate face");
        }
        used[static_cast<std::size_t>(a)] = 1;
        used[static_cast<std::size_t>(b)] = 1;
        used[static_cast<std::size_t>(c)] = 1;
        Eigen::RowVector3d const ab
            = vertices.row(b) - vertices.row(a);
        Eigen::RowVector3d const ac
            = vertices.row(c) - vertices.row(a);
        if (ab.cross(ac).squaredNorm() < area2_floor) {
            throw std::runtime_error(path + ": degenerate face");
        }
        auto add_edge = [&](int u, int v) {
            if (u > v) {
                std::swap(u, v);
            }
            int& count = edge_face_count[std::pair<int, int>(u, v)];
            ++count;
            if (count > 2) {
                throw std::runtime_error(path + ": non-manifold edge");
            }
        };
        add_edge(a, b);
        add_edge(b, c);
        add_edge(c, a);
    }
    for (char const seen : used) {
        if (seen == 0) {
            throw std::runtime_error(path + ": unused vertex");
        }
    }
}

inline void read_triangle_obj(
    std::string const& path,
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }

    std::vector<Eigen::RowVector3d> vertex_list;
    std::vector<Eigen::RowVector3i> face_list;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream stream(line);
        std::string type;
        stream >> type;
        if (type == "v") {
            Eigen::RowVector3d point;
            if (!(stream >> point(0) >> point(1) >> point(2))) {
                throw std::runtime_error(path + ": bad vertex");
            }
            vertex_list.push_back(point);
        } else if (type == "f") {
            Eigen::RowVector3i face;
            std::string token;
            int count = 0;
            while (stream >> token) {
                auto const slash = token.find('/');
                int index = 0;
                try {
                    index = std::stoi(token.substr(0, slash));
                } catch (std::exception const&) {
                    throw std::runtime_error(path + ": bad face index");
                }
                if (index < 1) {
                    throw std::runtime_error(path + ": face index out of range");
                }
                if (count >= 3) {
                    throw std::runtime_error(path + ": non-triangle face");
                }
                face(count) = index - 1;
                ++count;
            }
            if (count != 3) {
                throw std::runtime_error(path + ": non-triangle face");
            }
            face_list.push_back(face);
        }
    }
    if (vertex_list.empty() || face_list.empty()) {
        throw std::runtime_error(path + ": empty mesh");
    }

    vertices.resize(static_cast<int>(vertex_list.size()), 3);
    for (int i = 0; i < vertices.rows(); ++i) {
        vertices.row(i) = vertex_list[static_cast<std::size_t>(i)];
    }
    faces.resize(static_cast<int>(face_list.size()), 3);
    for (int i = 0; i < faces.rows(); ++i) {
        faces.row(i) = face_list[static_cast<std::size_t>(i)];
    }
    validate_triangle_mesh(vertices, faces, path);
}

inline void read_triangle_off(
    std::string const& path,
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }

    std::string magic;
    int vertex_count = 0;
    int face_count = 0;
    int edge_count = 0;
    if (!(file >> magic >> vertex_count >> face_count >> edge_count)
        || magic != "OFF" || vertex_count < 3 || face_count < 1) {
        throw std::runtime_error(path + ": bad OFF header");
    }

    vertices.resize(vertex_count, 3);
    for (int i = 0; i < vertex_count; ++i) {
        if (!(file >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2))) {
            throw std::runtime_error(path + ": bad vertex");
        }
    }

    faces.resize(face_count, 3);
    for (int i = 0; i < face_count; ++i) {
        int corners = 0;
        if (!(file >> corners) || corners != 3) {
            throw std::runtime_error(path + ": non-triangle face");
        }
        if (!(file >> faces(i, 0) >> faces(i, 1) >> faces(i, 2))) {
            throw std::runtime_error(path + ": bad face");
        }
    }
    validate_triangle_mesh(vertices, faces, path);
}

class WeavingMesh {
public:
    explicit WeavingMesh(std::string const& path)
    {
        Eigen::MatrixXd vertices;
        Eigen::MatrixXi faces;
        std::string const ext = mesh_extension(path);
        if (ext == ".obj") {
            read_triangle_obj(path, vertices, faces);
        } else if (ext == ".off") {
            read_triangle_off(path, vertices, faces);
        } else {
            throw std::runtime_error(path + ": expected .obj or .off");
        }
        mesh_.set_mesh(vertices, faces);
        tangent_bundle_.init(mesh_);
    }

    // Build from explicit geometry (used for the punctured sub-mesh).
    WeavingMesh(Eigen::MatrixXd const& vertices, Eigen::MatrixXi const& faces)
    {
        validate_triangle_mesh(vertices, faces, "punctured");
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
