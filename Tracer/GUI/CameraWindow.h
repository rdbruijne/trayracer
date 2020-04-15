#pragma once

// Project
#include "GuiWindow.h"

namespace Tracer
{
	struct CameraNode;
	class CameraWindow : public GuiWindow
	{
	public:
		static CameraWindow* const Get();

		CameraNode* mCamNode = nullptr;

	private:
		void DrawImpl() final;
	};
}
