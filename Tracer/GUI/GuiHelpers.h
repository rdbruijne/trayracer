#pragma once

#include <vector>

namespace Tracer
{
	struct CameraNode;
	class GuiWindow;
	class Window;
	class GuiHelpers
	{
	public:
		static bool Init(Window* window);
		static void DeInit();

		static void Draw();

		// app data
		static CameraNode* CamNode;

	private:
		static std::vector<GuiWindow*> msWindows;
	};
}
