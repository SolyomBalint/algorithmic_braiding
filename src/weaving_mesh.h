#pragma once

#include <directional/TriMesh.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/readOBJ.h>

#include <cstdlib>
#include <string>

class WeavingMesh {
public:
    explicit WeavingMesh(std::string const& path)
    {
        if (!directional::readOBJ(path, mesh_)) {
            std::abort();
        }
        tangent_bundle_.init(mesh_);
    }

    WeavingMesh(WeavingMesh const&) = delete;
    WeavingMesh& operator=(WeavingMesh const&) = delete;
    WeavingMesh(WeavingMesh&&) = delete;
    WeavingMesh& operator=(WeavingMesh&&) = delete;

    directional::TriMesh const& mesh() const { return mesh_; }
    directional::TriMesh& mesh() { return mesh_; }

    directional::PCFaceTangentBundle const& tangent_bundle() const
    {
        return tangent_bundle_;
    }

private:
    directional::TriMesh mesh_;
    directional::PCFaceTangentBundle tangent_bundle_;
};
