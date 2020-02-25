#pragma once

// Project
#include "OpenGL/Input.h"

namespace Tracer
{
	class Window;
	class ControlScheme
	{
	public:
		ControlScheme();

		// entry
		struct Entry
		{
			Entry() = default;
			explicit Entry(Input::Keys key, float scalar) : Entry(key, Input::ModifierKeys::None, scalar) {}
			explicit Entry(Input::Keys key, Input::ModifierKeys modifiers = Input::ModifierKeys::None, float scalar = 1.f);

			// mouse events will be based on scroll/move, keyboard will be 0 or 1 (-1 when inverted)
			float2 HandleInput(Window* window);

			Input::Keys Key = Input::Keys::Unknown;
			Input::ModifierKeys Modifiers = Input::ModifierKeys::None;
			float Scalar = 1.f;
		};

		// Entries
		Entry OrbitCameraMove;
		Entry OrbitCameraOrbit;
		Entry OrbitCameraRotate;
		Entry OrbitCameraRoll;
		Entry OrbitCameraDolly;
	};
}
