#pragma once

// Project
#include "GuiWindow.h"

namespace Tracer
{
	struct CameraNode;
	class CameraWindow : public GuiWindow
	{
	public:
		CameraNode* mCamNode = nullptr;

	private:
		void DrawImpl() final;
	};
}
