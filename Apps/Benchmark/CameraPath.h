#pragma once

// Tracer
#include "Tracer/Utility/LinearMath.h"
#include "Tracer/Resources/CameraNode.h"

// C++
#include <optional>
#include <string>
#include <vector>

namespace Benchmark
{
	class CameraPath
	{
	public:
		// nodes
		std::vector<Tracer::CameraNode>& Nodes() { return mNodes; }
		const std::vector<Tracer::CameraNode>& Nodes() const { return mNodes; }
		void SetNodes(const std::vector<Tracer::CameraNode>& nodes) { mNodes = nodes; }

		size_t NodeCount() const { return mNodes.size(); }
		Tracer::CameraNode Node(size_t index) { return mNodes[index]; }
		const Tracer::CameraNode& Node(size_t index) const { return mNodes[index]; }

		// node manipulation
		void Add(const Tracer::CameraNode& node, float time)
		{
			mNodes.push_back(node);
			mNodes.back().SetFlags(__float_as_uint(time));
		}

		// playback
		float TotalTime() const;
		std::optional<Tracer::CameraNode> Playback(float time);

		// load/save
		bool Load(const std::string& filepath);
		bool Save(const std::string& filepath);

	private:
		std::vector<Tracer::CameraNode> mNodes;
	};
}
