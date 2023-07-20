#pragma once

// Tracer
#include "OpenGL/Input.h"
#include "Utility/LinearMath.h"

namespace Tracer
{
	class CameraNode;
	class Window;
	class FlyCameraController
	{
	public:
		static bool HandleInput(CameraNode& node, Window* window, float dt);

	private:
		// keybinds
		inline static Input::Keybind mMoveForward = Input::Keybind(Input::Keys::W);
		inline static Input::Keybind mMoveBack    = Input::Keybind(Input::Keys::S);
		inline static Input::Keybind mMoveLeft    = Input::Keybind(Input::Keys::A);
		inline static Input::Keybind mMoveRight   = Input::Keybind(Input::Keys::D);
		inline static Input::Keybind mMoveUp      = Input::Keybind(Input::Keys::R);
		inline static Input::Keybind mMoveDown    = Input::Keybind(Input::Keys::F);

		inline static Input::Keybind mTiltUp      = Input::Keybind(Input::Keys::Up);
		inline static Input::Keybind mTiltDown    = Input::Keybind(Input::Keys::Down);
		inline static Input::Keybind mPanLeft     = Input::Keybind(Input::Keys::Left);
		inline static Input::Keybind mPanRight    = Input::Keybind(Input::Keys::Right);
		inline static Input::Keybind mRollLeft    = Input::Keybind(Input::Keys::Q);
		inline static Input::Keybind mRollRight   = Input::Keybind(Input::Keys::E);
	};
}