#include "ControlScheme.h"

// Tracer
#include "Tracer/OpenGL/Window.h"

using namespace Tracer;
namespace Demo
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



	ControlScheme::Entry::Entry(Input::Keys key) :
		Entry(key, Tracer::Input::ModifierKeys::None, 1.f)
	{
	}



	ControlScheme::Entry::Entry(Input::Keys key, float scalar) :
		Entry(key, Tracer::Input::ModifierKeys::None, scalar)
	{
	}



	ControlScheme::Entry::Entry(Input::Keys key, Input::ModifierKeys modifiers, float scalar) :
		mKey(key),
		mModifiers(modifiers),
		mScalar(scalar)
	{
	}



	float2 ControlScheme::Entry::HandleInput(Window* window)
	{
		// handle scrollwheel
		if(mKey == Input::Keys::Mouse_Scroll)
			return window->ScrollDelta() * mScalar;

		// check key
		if(!window->IsKeyDown(mKey))
			return make_float2(0, 0);

		// check modifier keys
		if(!CheckModifier(mModifiers, window))
			return make_float2(0, 0);

		// return scroll if the key is a mouse button
		if(Input::IsMouseKey(mKey))
			return window->CursorDelta() * mScalar;

		// return keyboard result
		return make_float2(mScalar, mScalar);
	}
}
