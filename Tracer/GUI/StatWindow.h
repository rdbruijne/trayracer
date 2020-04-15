#pragma once

// Project
#include "GuiWindow.h"

// C++
#include <stdint.h>

namespace Tracer
{
	class Renderer;
	class StatWindow : public GuiWindow
	{
	public:
		static StatWindow* const Get();

		int64_t mFrameTimeNs = 0;
		Renderer* mRenderer = nullptr;

	private:
		void DrawImpl() final;
	};
}
