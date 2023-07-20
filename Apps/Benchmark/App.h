#pragma once

// Project
#include "CameraPath.h"

// Tracer
#include "Tracer/App/App.h"
#include "Tracer/Renderer/RenderStatistics.h"

namespace Benchmark
{
	class App : public Tracer::App
	{
		// disable copying
		App(const App&) = delete;
		App& operator =(const App&) = delete;

		// disable moving
		App(const App&&) = delete;
		App& operator =(const App&&) = delete;

	public:
		App() = default;

		void Init(Tracer::Renderer* renderer, Tracer::Window* window) override;
		void DeInit(Tracer::Renderer* renderer, Tracer::Window* window) override;
		void Tick(Tracer::Renderer* renderer, Tracer::Window* window, float dt) override;

	private:
		float mPlayTime = -1;
		CameraPath mCameraPath;

		// benchmark data
		int mBenchFrameCount = 0;
		float mBenchTime = 0;
		Tracer::RenderStatistics mBenchStats;
	};
}
