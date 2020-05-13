#pragma once

// Project
#include "BaseGui.h"

namespace Tracer
{
	class CameraNode;
	class Renderer;
	class Scene;
	class Window;
	class RendererGui : public BaseGui
	{
	public:
		static RendererGui* const Get();

		CameraNode* mCamNode = nullptr;
		Renderer* mRenderer = nullptr;
		Scene* mScene = nullptr;
		Window* mWindow = nullptr;

	private:
		void DrawImpl() final;
	};
}
