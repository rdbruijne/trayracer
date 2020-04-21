#pragma once

// Project
#include "GuiWindow.h"

namespace Tracer
{
	class CameraNode;
	class CameraWindow : public GuiWindow
	{
	public:
		static CameraWindow* const Get();

		CameraNode* mCamNode = nullptr;

	private:
		void DrawImpl() final;
	};
}
