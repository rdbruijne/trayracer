#pragma once

// Project
#include "OpenGL/Input.h"

// C++
#include <vector>

namespace Tracer
{
	struct CameraNode;
	class GuiWindow;
	class Renderer;
	class Window;
	class GuiHelpers
	{
	public:
		static bool Init(Window* window);
		static void DeInit();

		static void BeginFrame();
		static void EndFrame();
	};
}
