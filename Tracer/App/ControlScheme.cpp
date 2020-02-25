#include "App/ControlScheme.h"

// Project
#include "OpenGL/Window.h"

namespace Tracer
{
	namespace
	{
		inline bool CheckModifier(Input::ModifierKeys modifier, Window* window)
		{
			// check keys
			const bool altDown = window->IsKeyDown(Input::Keys::LeftAlt) || window->IsKeyDown(Input::Keys::RightAlt);
			const bool ctrlDown = window->IsKeyDown(Input::Keys::LeftControl) || window->IsKeyDown(Input::Keys::RightControl);
			const bool shiftDown = window->IsKeyDown(Input::Keys::LeftShift) || window->IsKeyDown(Input::Keys::RightShift);

			// apply keys
			bool result = true;
			result = result && ((modifier & Input::ModifierKeys::Alt) == Input::ModifierKeys::Alt) == altDown;
			result = result && ((modifier & Input::ModifierKeys::Ctrl) == Input::ModifierKeys::Ctrl) == ctrlDown;
			result = result && ((modifier & Input::ModifierKeys::Shift) == Input::ModifierKeys::Shift) == shiftDown;
			return result;
		}
	}



	ControlScheme::ControlScheme()
	{
		// load defaults
		OrbitCameraMove   = Entry(Input::Keys::Mouse_Right, .05f);
		OrbitCameraOrbit  = Entry(Input::Keys::Mouse_Left, .005f);
		OrbitCameraRotate = Entry(Input::Keys::Mouse_Left, Input::ModifierKeys::Alt, .005f);
		OrbitCameraRoll   = Entry(Input::Keys::Mouse_Right, Input::ModifierKeys::Alt, .01f);
		OrbitCameraDolly  = Entry(Input::Keys::Mouse_Middle, .01f);
	}



	ControlScheme::Entry::Entry(Input::Keys key, Input::ModifierKeys modifiers, float scalar) :
		Key(key),
		Modifiers(modifiers),
		Scalar(scalar)
	{
	}



	float2 ControlScheme::Entry::HandleInput(Window* window)
	{
		// check key
		if(!window->IsKeyDown(Key))
			return make_float2(0, 0);

		// check modifier keys
		if(!CheckModifier(Modifiers, window))
			return make_float2(0, 0);

		// return scroll if the key is a mouse button
		if(Key >= Input::Keys::_FirstMouse && Key <= Input::Keys::_LastMouse)
		{
#if 0
			return window->GetCursorDelta() * Scalar;
#else
			const float2 delta = window->GetCursorDelta();
			return delta * Scalar;
#endif
		}

		// return keyboard result
		return make_float2(Scalar, Scalar);
	}
}
