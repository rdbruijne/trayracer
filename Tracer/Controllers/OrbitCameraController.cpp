#include "OrbitCameraController.h"

// Tracer
#include "OpenGL/Window.h"
#include "Resources/CameraNode.h"

namespace Tracer
{
	bool OrbitCameraController::HandleInput(CameraNode& node, Window* window)
	{
		// check inputs
		const float2 camMove   = window->CheckInput(sOrbitCameraMove);
		const float2 camOrbit  = window->CheckInput(sOrbitCameraOrbit);
		const float2 camRotate = window->CheckInput(sOrbitCameraRotate);
		const float2 camRoll   = window->CheckInput(sOrbitCameraRoll);
		const float2 camDolly  = window->CheckInput(sOrbitCameraDolly) + window->CheckInput(sOrbitCameraDollyAlt);

		// apply inputs
		bool hasChanged = false;
		hasChanged = OrbitCamera(node, -camOrbit) || hasChanged;
		hasChanged = PanCamera(node, camMove * make_float2(-1, 1)) || hasChanged;
		hasChanged = DollyCamera(node, camDolly.y) || hasChanged;
		hasChanged = RotateCamera(node, camRotate.y, camRotate.x, 0) || hasChanged;
		sPrevUp = RecalculateUpVector(node, sPrevUp);
		return hasChanged;
	}



	float3 OrbitCameraController::RecalculateUpVector(CameraNode& node, const float3& prevUp)
	{
		const float3 dir = normalize(node.Target() - node.Position());
		float3 up = node.Up() - dir * dot(node.Up(), dir);

		// up is parallel to dir
		constexpr float epsilonSqr = Epsilon * Epsilon;
		if(dot(up, up) < (epsilonSqr))
		{
			// try old up
			up = prevUp - dir * dot(prevUp, dir);

			// up is still parallel to dir
			if(dot(up, up) < (epsilonSqr))
			{
				up = SmallestAxis(dir);
				up = up - dir * dot(up, dir);
			}
		}

		return normalize(up);
	}



	bool OrbitCameraController::DollyCamera(CameraNode& node, float dolly)
	{
		if(dolly == 0)
			return false;

		node.SetPosition(node.Target() + ((node.Position() - node.Target()) * expf(dolly)));
		return true;
	}



	bool OrbitCameraController::MoveCamera(CameraNode& node, const float3& move)
	{
		if(move.x == 0 && move.y == 0 && move.z == 0)
			return false;

		const float3 dir = node.Target() - node.Position();
		const float3 diff = ((normalize(cross(dir, sPrevUp)) * move.x) + (sPrevUp * move.y)) * length(dir) + (dir * move.z);

		node.SetPosition(node.Position() + diff);
		node.SetTarget(node.Target() + diff);

		return true;
	}



	bool OrbitCameraController::OrbitCamera(CameraNode& node, const float2& orbit)
	{
		if(orbit.x == 0 || orbit.y == 0)
			return false;

		float3 dir = node.Target() - node.Position();
		const float3 up = normalize(node.Up());
		const float3 side = normalize(cross(dir, sPrevUp));

		// up/down
		const float3 cachedUp = sPrevUp;
		sPrevUp = RotateAroundAxis(sPrevUp, side, orbit.y);

		// clamp
		constexpr float minDot = 1e-3f;
		float y = orbit.y;
		float dotUp = dot(cachedUp, up);
		if(dot(sPrevUp, up) < minDot)
		{
			sPrevUp = cachedUp;
			if(dotUp < minDot)
			{
				y = 0;
			}
			else
			{
				const float a = acosf(minDot) - acosf(dotUp);
				y = orbit.y < 0 ? -a : a;
				sPrevUp = RotateAroundAxis(sPrevUp, side, y);
			}
		}

		dir = RotateAroundAxis(dir, side, y);

		// left/right
		sPrevUp = RotateAroundAxis(sPrevUp, up, orbit.x);
		dir = RotateAroundAxis(dir, up, orbit.x);

		node.SetPosition(node.Target() - dir);
		return true;
	}



	bool OrbitCameraController::PanCamera(CameraNode& node, const float2& pan)
	{
		if(pan.x == 0 && pan.y == 0)
			return false;

		const float3 dir = node.Target() - node.Position();
		const float3 side = normalize(cross(dir, sPrevUp));
		const float dst = length(dir);
		const float3 diff = ((side * dst * pan.x) + (sPrevUp * dst * pan.y)) * tanf(node.Fov() * DegToRad * .5f) * 2.8f;

		node.SetPosition(node.Position() + diff);
		node.SetTarget(node.Target() + diff);

		return true;
	}



	bool OrbitCameraController::RotateCamera(CameraNode& node, float tilt, float pan, float roll)
	{
		if(tilt == 0 && pan == 0 && roll == 0)
			return false;

		float3 dir = node.Target() - node.Position();
		const float3 side = normalize(cross(dir, sPrevUp));

		// tilt
		dir = RotateAroundAxis(dir, side, tilt);
		sPrevUp = RotateAroundAxis(sPrevUp, side, tilt);

		// pan
		dir = RotateAroundAxis(dir, sPrevUp, pan);

		// roll
		sPrevUp = RotateAroundAxis(sPrevUp, normalize(dir), roll);

		node.SetTarget(node.Position() + dir);

		// only update up vector when rolling
		if(roll != 0)
			node.SetUp(sPrevUp);

		return true;
	}
}
