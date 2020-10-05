#pragma once

// Tracer
#include "Tracer/Utility/LinearMath.h"

namespace Tracer
{
	class CameraNode;
	class Window;
}

namespace Demo
{
	class ControlScheme;
	class OrbitCameraController
	{
	public:
		static bool HandleInput(Tracer::CameraNode& node, ControlScheme* scheme, Tracer::Window* window);
		static float3 RecalculateUpVector(Tracer::CameraNode& node, const float3& prevUp);

	private:
		static bool DollyCamera(Tracer::CameraNode& node, float strafe);
		static bool MoveCamera(Tracer::CameraNode& node, const float3& move);
		static bool OrbitCamera(Tracer::CameraNode& node, const float2& orbit);
		static bool PanCamera(Tracer::CameraNode& node, const float2& pan);
		static bool RotateCamera(Tracer::CameraNode& node, float tilt, float pan, float roll);


		static float3 sPrevUp;
	};
}
