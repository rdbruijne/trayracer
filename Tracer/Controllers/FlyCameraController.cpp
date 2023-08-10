#include "FlyCameraController.h"

// Project
#include "OpenGL/Window.h"
#include "Resources/CameraNode.h"

namespace Tracer
{
	bool FlyCameraController::HandleInput(CameraNode& node, Window* window, float dt)
	{
		// check inputs
		const float moveX = window->CheckInput(mMoveRight) - window->CheckInput(mMoveLeft);
		const float moveY = window->CheckInput(mMoveUp) - window->CheckInput(mMoveDown);
		const float moveZ = window->CheckInput(mMoveForward) - window->CheckInput(mMoveBack);

		const float tilt = window->CheckInput(mTiltDown) - window->CheckInput(mTiltUp);
		const float pan = window->CheckInput(mPanLeft) - window->CheckInput(mPanRight);
		const float roll = window->CheckInput(mRollRight) - window->CheckInput(mRollLeft);

		// early-out if no input detected
		if(moveX == 0 && moveY == 0 && moveZ == 0 && tilt == 0 && pan == 0 && roll == 0)
			return false;

		// multipliers
		float moveScalar = 100.f * dt;
		float rotateScalar = .5f * dt;
		constexpr float moveMultiplier = 10.f;
		constexpr float rotateMultiplier = 10.f;

		if(window->IsDown(Input::Modifiers::Shift))
		{
			moveScalar *= moveMultiplier;
			rotateScalar *= rotateMultiplier;
		}

		if(window->IsDown(Input::Modifiers::Ctrl))
		{
			moveScalar /= moveMultiplier;
			rotateScalar /= rotateMultiplier;
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
