#include "FlyCameraController.h"

// Tracer
#include "Tracer/OpenGL/Window.h"
#include "Tracer/Resources/CameraNode.h"

using namespace Tracer;

namespace Benchmark
{
	bool FlyCameraController::HandleInput(CameraNode& node, Window* window, float dt)
	{
		// check inputs
		const float moveX = window->CheckInput(mMoveRight).x - window->CheckInput(mMoveLeft).x;
		const float moveY = window->CheckInput(mMoveUp).x - window->CheckInput(mMoveDown).x;
		const float moveZ = window->CheckInput(mMoveForward).x - window->CheckInput(mMoveBack).x;

		const float tilt = window->CheckInput(mTiltDown).x - window->CheckInput(mTiltUp).x;
		const float pan = window->CheckInput(mPanLeft).x - window->CheckInput(mPanRight).x;
		const float roll = window->CheckInput(mRollRight).x - window->CheckInput(mRollLeft).x;

		// early-out if no input detected
		if(moveX == 0 && moveY == 0 && moveZ == 0 && tilt == 0 && pan == 0 && roll == 0)
			return false;

		// multipliers
		float moveScalar = 100.f * dt;
		float rotateScalar = .5f * dt;
		constexpr float moveMultiplier = 10.f;
		constexpr float RotateMultiplier = 10.f;

		if(window->IsModifierDown(Input::ModifierKeys::Shift))
		{
			moveScalar *= moveMultiplier;
			rotateScalar *= RotateMultiplier;
		}

		if(window->IsModifierDown(Input::ModifierKeys::Ctrl))
		{
			moveScalar /= moveMultiplier;
			rotateScalar /= RotateMultiplier;
		}



		// camera transform
		const float3x4 camT = node.Transform();

		// create transform
		const float3x4 rotate = rotate_3x4(tilt * rotateScalar, pan * rotateScalar, roll * rotateScalar);
		const float3x4 camT2 = rotate * camT;
		const float3 translate = ((camT.x * moveX) + (camT.y * moveY) + (camT.z * moveZ)) * moveScalar;

		// apply to camera
		node.SetPosition(node.Position() + translate);
		node.SetUp(camT2.y);
		node.SetTarget(node.Position() + camT2.z);

		return true;
	}
}
