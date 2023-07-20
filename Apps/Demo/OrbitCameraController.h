#pragma once

// Tracer
#include "Tracer/OpenGL/Input.h"
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
		static bool HandleInput(Tracer::CameraNode& node, Tracer::Window* window);

	private:
		static float3 RecalculateUpVector(Tracer::CameraNode& node, const float3& prevUp);

		static bool DollyCamera(Tracer::CameraNode& node, float strafe);
		static bool MoveCamera(Tracer::CameraNode& node, const float3& move);
		static bool OrbitCamera(Tracer::CameraNode& node, const float2& orbit);
		static bool PanCamera(Tracer::CameraNode& node, const float2& pan);
		static bool RotateCamera(Tracer::CameraNode& node, float tilt, float pan, float roll);

		inline static float3 sPrevUp = make_float3(0, 1, 0);

		// keybinds
		using Keys = Tracer::Input::Keys;
		using ModifierKeys = Tracer::Input::ModifierKeys;
		using Keybind = Tracer::Input::Keybind;

		inline static Keybind sOrbitCameraMove     = Keybind(Keys::Mouse_Middle, .05f);
		inline static Keybind sOrbitCameraOrbit    = Keybind(Keys::Mouse_Left, .005f);
		inline static Keybind sOrbitCameraRotate   = Keybind(Keys::Mouse_Left, ModifierKeys::Alt, .005f);
		inline static Keybind sOrbitCameraRoll     = Keybind(Keys::Mouse_Right, ModifierKeys::Alt, .01f);
		inline static Keybind sOrbitCameraDolly    = Keybind(Keys::Mouse_Right, .01f);
		inline static Keybind sOrbitCameraDollyAlt = Keybind(Keys::Mouse_Scroll, -.1f);
	};
}
