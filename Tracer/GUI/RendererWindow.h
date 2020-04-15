#pragma once

// Project
#include "GuiWindow.h"

namespace Tracer
{
	class Renderer;
	class RendererWindow : public GuiWindow
	{
	public:
		Renderer* mRenderer = nullptr;

	private:
		void DrawImpl() final;
	};
}
