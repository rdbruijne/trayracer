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
		inline static Input::Keybind sOrbitCameraMove     = Input::Keybind(Input::Keys::Mouse_Middle, .05f);
		inline static Input::Keybind sOrbitCameraOrbit    = Input::Keybind(Input::Keys::Mouse_Left, .005f);
		inline static Input::Keybind sOrbitCameraRotate   = Input::Keybind(Input::Keys::Mouse_Left, Input::ModifierKeys::Alt, .005f);
		inline static Input::Keybind sOrbitCameraRoll     = Input::Keybind(Input::Keys::Mouse_Right, Input::ModifierKeys::Alt, .01f);
		inline static Input::Keybind sOrbitCameraDolly    = Input::Keybind(Input::Keys::Mouse_Right, .01f);
		inline static Input::Keybind sOrbitCameraDollyAlt = Input::Keybind(Input::Keys::Mouse_Scroll, -.1f);
	};
}
