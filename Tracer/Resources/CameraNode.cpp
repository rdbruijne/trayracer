#include "CameraNode.h"

namespace Tracer
{
	CameraNode::CameraNode(const float3& position, const float3& target, const float3& up, float fov) :
		Resource(""),
		mPosition(position),
		mTarget(target),
		mUp(normalize(up)),
		mFov(fov)
	{
	}



	void CameraNode::SetPosition(const float3& position)
	{
		if(mPosition != position)
		{
			mPosition = position;
			MarkDirty();
		}
	}



	void CameraNode::SetTarget(const float3& target)
	{
		if(mTarget != target)
		{
			mTarget = target;
			MarkDirty();
		}
	}



	void CameraNode::SetUp(const float3& up)
	{
		if(mUp != up)
		{
			mUp = normalize(up);
			MarkDirty();
		}
	}



	float3x4 CameraNode::Transform() const
	{
		//const float3 fwd = normalize(mTarget - mPosition);
		//const float3 side = cross(fwd, mUp);
		//const float3 up = cross(side, fwd);
		//const float3 pos = mPosition;
		//return make_float3x4(side, up, fwd, pos);

		const float3 pos = mPosition;
		const float3 fwd = normalize(mTarget - mPosition);
		const float3 side = normalize(cross(fwd, mUp));
		const float3 up = normalize(cross(side, fwd));
		return make_float3x4(side, up, fwd, pos);
	}



	void CameraNode::SetAperture(float aperture)
	{
		if(mAperture != aperture)
		{
			mAperture = aperture;
			MarkDirty();
		}
	}



	void CameraNode::SetDistortion(float distortion)
	{
		if(mDistortion != distortion)
		{
			mDistortion = distortion;
			MarkDirty();
		}
	}



	void CameraNode::SetFocalDist(float dist)
	{
		if(mFocalDist != dist)
		{
			mFocalDist = dist;
			MarkDirty();
		}
	}



	void CameraNode::SetFov(float fov)
	{
		if(mFov != fov)
		{
			mFov = fov;
			MarkDirty();
		}
	}



	void CameraNode::SetBokehSideCount(int count)
	{
		if(count != mBokehSideCount)
		{
			mBokehSideCount = count;
			MarkDirty();
		}
	}



	void CameraNode::SetBokehRotation(float rotation)
	{
		if(mBokehRotation != rotation)
		{
			mBokehRotation = rotation;
			MarkDirty();
		}
	}
}
