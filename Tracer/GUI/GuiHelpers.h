#pragma once

// Project
#include "OpenGL/Input.h"

// C++
#include <vector>

namespace Tracer
{
	class CameraNode;
	class Renderer;
	class Scene;
	class Window;
	class GuiHelpers
	{
	public:
		static bool Init(Window* renderWindow);
		static void DeInit();

		static void BeginFrame();
		static void EndFrame();

		static inline CameraNode* camNode = nullptr;
		static inline Renderer* renderer = nullptr;
		static inline Scene* scene = nullptr;
		static inline Window* window = nullptr;
	};
}
