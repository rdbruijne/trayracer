#pragma once

// Project
#include "BaseGui.h"

namespace Tracer
{
	class RendererGui : public BaseGui
	{
	public:
		static RendererGui* const Get();

	private:
		void DrawImpl() final;
	};
}
