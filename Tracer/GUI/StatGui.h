#pragma once

// Project
#include "BaseGui.h"

// C++
#include <stdint.h>

namespace Tracer
{
	class Renderer;
	class Scene;
	class StatGui : public BaseGui
	{
	public:
		static StatGui* const Get();

		float mFrameTimeMs = 0;
		float mBuildTimeMs = 0;

	private:
		void DrawImpl() final;
	};
}
