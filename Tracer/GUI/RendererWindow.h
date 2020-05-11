#pragma once

// Project
#include "GuiWindow.h"

namespace Tracer
{
	class Renderer;
	class Window;
	class RendererWindow : public GuiWindow
	{
	public:
		static RendererWindow* const Get();

		Renderer* mRenderer = nullptr;
		Window* mWindow = nullptr;

	private:
		void DrawImpl() final;
	};
}
