#pragma once

// Project
#include "OpenGL/Input.h"

// C++
#include <vector>

namespace Tracer
{
	class BaseGui;
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
