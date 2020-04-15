#pragma once

// Project
#include "Utility/LinearMath.h"

namespace Tracer
{
	struct CameraNode;
	class ControlScheme;
	class Window;
	class OrbitCameraController
	{
	public:
		static bool HandleInput(CameraNode& node, ControlScheme* scheme, Window* window);
		static float3 RecalculateUpVector(CameraNode& node, const float3& prevUp);

	private:
		static bool DollyCamera(CameraNode& node, float strafe);
		static bool MoveCamera(CameraNode& node, const float3& move);
		static bool OrbitCamera(CameraNode& node, const float2& orbit);
		static bool PanCamera(CameraNode& node, const float2& pan);
		static bool RotateCamera(CameraNode& node, float tilt, float pan, float roll);


		static float3 sPrevUp;
	};
}
