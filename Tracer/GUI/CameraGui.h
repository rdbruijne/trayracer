#pragma once

// Project
#include "BaseGui.h"

namespace Tracer
{
	class CameraNode;
	class CameraGui : public BaseGui
	{
	public:
		static CameraGui* const Get();

		CameraNode* mCamNode = nullptr;

	private:
		void DrawImpl() final;
	};
}
