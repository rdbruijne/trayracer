#pragma once

// Project
#include "GuiWindow.h"

namespace Tracer
{
	class Renderer;
	class RendererWindow : public GuiWindow
	{
	public:
		static RendererWindow* const Get();

		Renderer* mRenderer = nullptr;

	private:
		void DrawImpl() final;
	};
}
