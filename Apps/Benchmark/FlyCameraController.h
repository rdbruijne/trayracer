#pragma once

// Tracer
#include "Tracer/OpenGL/Input.h"
#include "Tracer/Utility/LinearMath.h"

namespace Tracer
{
	class CameraNode;
	class Window;
}

namespace Benchmark
{
	class FlyCameraController
	{
	public:
		static bool HandleInput(Tracer::CameraNode& node, Tracer::Window* window, float dt);

	private:
		// keybinds
		using Keys = Tracer::Input::Keys;
		using Keybind = Tracer::Input::Keybind;

		inline static Keybind mMoveForward = Keybind(Keys::W);
		inline static Keybind mMoveBack    = Keybind(Keys::S);
		inline static Keybind mMoveLeft    = Keybind(Keys::A);
		inline static Keybind mMoveRight   = Keybind(Keys::D);
		inline static Keybind mMoveUp      = Keybind(Keys::R);
		inline static Keybind mMoveDown    = Keybind(Keys::F);

		inline static Keybind mTiltUp      = Keybind(Keys::Up);
		inline static Keybind mTiltDown    = Keybind(Keys::Down);
		inline static Keybind mPanLeft     = Keybind(Keys::Left);
		inline static Keybind mPanRight    = Keybind(Keys::Right);
		inline static Keybind mRollLeft    = Keybind(Keys::Q);
		inline static Keybind mRollRight   = Keybind(Keys::E);
	};
}
