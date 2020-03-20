#pragma once

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

		static void Draw();

		// app data
		static inline CameraNode* camNode = nullptr;
		static inline Renderer* renderer = nullptr;

	private:
		static inline std::vector<GuiWindow*> msWindows = {};
	};
}
