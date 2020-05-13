#pragma once

// Project
#include "BaseGui.h"

namespace Tracer
{
	class Renderer;
	class Window;
	class RendererGui : public BaseGui
	{
	public:
		static RendererGui* const Get();

		Renderer* mRenderer = nullptr;
		Window* mWindow = nullptr;

	private:
		void DrawImpl() final;
	};
}
