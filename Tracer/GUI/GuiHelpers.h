#pragma once

// Project
#include "GUI/GuiWindow.h"
#include "OpenGL/Input.h"

// C++
#include <map>

namespace Tracer
{
	class CameraNode;
	class Renderer;
	class Scene;
	class Window;
	class GuiHelpers
	{
	public:
		// initializations
		static void Init(Window* renderWindow);
		static void DeInit();

		// drawing
		static void BeginFrame();
		static void EndFrame();
		static void DrawGui();

		// registering
		template<class Type>
		static void Register(Input::Keys key)
		{
			mGuiItems[key] = GuiWindow::Get<Type>();
		}

		// set data
		static void Set(CameraNode* node) { mCamNode = node; }
		static void Set(Renderer* renderer) { mRenderer = renderer; }
		static void Set(Scene* scene) { mScene = scene; }
		static void SetFrameTimeMs(float frameTimeMs) { mFrameTimeMs = frameTimeMs; }

		// get data
		static CameraNode* GetCamNode() { return mCamNode; }
		static Renderer* GetRenderer() { return mRenderer; }
		static float GetFrameTimeMs() { return mFrameTimeMs; }
		static Window* GetRenderWindow() { return mRenderWindow; }
		static Scene* GetScene() { return mScene; }

	private:
		// data
		static inline CameraNode* mCamNode = nullptr;
		static inline float mFrameTimeMs = 0;
		static inline Renderer* mRenderer = nullptr;
		static inline Window* mRenderWindow = nullptr;
		static inline Scene* mScene = nullptr;

		// gui windows
		static inline std::map<Input::Keys, GuiWindow*> mGuiItems;
	};
}
