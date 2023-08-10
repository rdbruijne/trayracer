#pragma once

// Project
#include "OpenGL/Input.h"
#include "Utility/LinearMath.h"

namespace Tracer
{
	class CameraNode;
	class Window;
	class OrbitCameraController
	{
	public:
		static bool HandleInput(CameraNode& node, Window* window);

	private:
		static float3 RecalculateUpVector(CameraNode& node, const float3& prevUp);

		static bool DollyCamera(CameraNode& node, float strafe);
		static bool MoveCamera(CameraNode& node, const float3& move);
		static bool OrbitCamera(CameraNode& node, const float2& orbit);
		static bool PanCamera(CameraNode& node, const float2& pan);
		static bool RotateCamera(CameraNode& node, float tilt, float pan, float roll);

		inline static float3 sPrevUp = make_float3(0, 1, 0);

		// keybinds
		inline static float sMoveSpeed     = .05f;
		inline static float sOrbitSpeed    = .005f;
		inline static float sRotateSpeed   = .005f;
		inline static float sRollSpeed     = .01f;
		inline static float sDollySpeed    = .01f;
		inline static float sDollyAltSpeed = -.1f;

		inline static Input::Keybind sMove     = Input::Keybind(Input::MouseButtons::Middle);
		inline static Input::Keybind sOrbit    = Input::Keybind(Input::MouseButtons::Left);
		inline static Input::Keybind sRotate   = Input::Keybind(Input::MouseButtons::Left, Input::Modifiers::Alt);
		//inline static Input::Keybind sRoll     = Input::Keybind(Input::MouseButtons::Right, Input::Modifiers::Alt);
		inline static Input::Keybind sDolly    = Input::Keybind(Input::MouseButtons::Right);
		inline static Input::Keybind sDollyAlt = Input::Keybind(Input::MouseScroll::Vertical);
	};
}
