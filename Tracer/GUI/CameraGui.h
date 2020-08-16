#pragma once

// Project
#include "BaseGui.h"

namespace Tracer
{
	class CameraGui : public BaseGui
	{
	public:
		static CameraGui* const Get();

	private:
		void DrawImpl() final;
	};
}
