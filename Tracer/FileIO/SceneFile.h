#pragma once

// C++
#include <string>

namespace Tracer
{
	class CameraNode;
	class Renderer;
	class Scene;
	class Sky;
	class Window;
	class SceneFile
	{
	public:
		static void Load(const std::string& sceneFile, Scene* scene, Sky* sky, CameraNode* camNode, Renderer* renderer, Window* window);
		static void Save(const std::string& sceneFile, Scene* scene, Sky* sky, CameraNode* camNode, Renderer* renderer, Window* window);
	};
}
