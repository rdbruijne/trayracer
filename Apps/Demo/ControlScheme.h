#pragma once

// Tracer
#include "Tracer/OpenGL/Input.h"

namespace Tracer
{
	class Window;
}

namespace Demo
{
	class ControlScheme
	{
	public:
		ControlScheme();

		// entry
		struct Entry
		{
			Entry() = default;
			explicit Entry(Tracer::Input::Keys key, float scalar);
			explicit Entry(Tracer::Input::Keys key, Tracer::Input::ModifierKeys modifiers = Tracer::Input::ModifierKeys::None, float scalar = 1.f);

			// mouse events will be based on scroll/move, keyboard will be 0 or 1
			float2 HandleInput(Tracer::Window* window);

		private:
			Tracer::Input::Keys mKey = Tracer::Input::Keys::Unknown;
			Tracer::Input::ModifierKeys mModifiers = Tracer::Input::ModifierKeys::None;

			float mScalar = 1.f;
		};

		// Entries
		Entry OrbitCameraMove;
		Entry OrbitCameraOrbit;
		Entry OrbitCameraRotate;
		Entry OrbitCameraRoll;
		Entry OrbitCameraDolly;
		Entry OrbitCameraDollyAlt;
	};
}
