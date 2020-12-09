#include "CameraNode.h"

namespace Tracer
{
	CameraNode::CameraNode(const float3& position, const float3& target, const float3& up, float fov) :
		Resource(""),
		mPosition(position),
		mTarget(target),
		mUp(up),
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
			mUp = up;
			MarkDirty();
		}
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
}
