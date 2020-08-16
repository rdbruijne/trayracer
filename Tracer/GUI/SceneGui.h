#pragma once

// Project
#include "BaseGui.h"

namespace Tracer
{
	class SceneGui : public BaseGui
	{
	public:
		static SceneGui* const Get();

	private:
		void DrawImpl() final;

		void DrawScene();
		void DrawModels();
		void DrawInstances();

		void SelectModel(int ix);
		void SelectInstance(int ix);

		// generic
		static constexpr int mNameBufferSize = 128;

		// models
		int mSelectedModelIx = 0;
		char mModelName[mNameBufferSize];

		// instances
		int mSelectedInstanceIx = 0;
		char mInstanceName[mNameBufferSize];
	};
}
