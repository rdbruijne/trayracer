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
		// Entry
		struct Entry
		{
			Entry() = default;
			explicit Entry(Tracer::Input::Keys key);
			explicit Entry(Tracer::Input::Keys key, float scalar);
			explicit Entry(Tracer::Input::Keys key, Tracer::Input::ModifierKeys modifiers, float scalar);

			// mouse events will be based on scroll/move, keyboard will be 0 or 1
			float2 HandleInput(Tracer::Window* window);

		private:
			Tracer::Input::Keys mKey = Tracer::Input::Keys::Unknown;
			Tracer::Input::ModifierKeys mModifiers = Tracer::Input::ModifierKeys::None;

			float mScalar = 1.f;
		};

		// Entries
		Entry OrbitCameraMove     = Entry(Tracer::Input::Keys::Mouse_Middle, .05f);
		Entry OrbitCameraOrbit    = Entry(Tracer::Input::Keys::Mouse_Left, .005f);
		Entry OrbitCameraRotate   = Entry(Tracer::Input::Keys::Mouse_Left, Tracer::Input::ModifierKeys::Alt, .005f);
		Entry OrbitCameraRoll     = Entry(Tracer::Input::Keys::Mouse_Right, Tracer::Input::ModifierKeys::Alt, .01f);
		Entry OrbitCameraDolly    = Entry(Tracer::Input::Keys::Mouse_Right, .01f);
		Entry OrbitCameraDollyAlt = Entry(Tracer::Input::Keys::Mouse_Scroll, -.1f);
	};
}
