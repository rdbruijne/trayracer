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
		static bool Load(const std::string& sceneFile, Scene* scene, Sky* sky, CameraNode* camNode, Renderer* renderer, Window* window);
		static bool Save(const std::string& sceneFile, Scene* scene, Sky* sky, CameraNode* camNode, Renderer* renderer, Window* window);
	};
}
