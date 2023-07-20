#pragma once

// Tracer
#include "Tracer/GUI/GuiWindow.h"
#include "Tracer/Renderer/RenderStatistics.h"

namespace Benchmark
{
	class CameraPath;
	class RecorderGui : public Tracer::GuiWindow
	{
	public:
		CameraPath* Path() { return mPath; }
		void SetPath(CameraPath* path) { mPath = path; }

		// playback
		bool StartPlayback() { return mStartPlayback; }
		bool StopPlayback() { return mStopPlayback; }
		void SetPlayTime(float t) { mPlayTime = t; }

		// benchmark result data
		void SetBenchmarkResult(int frameCount, float elapsedMS, const Tracer::RenderStatistics& stats)
		{
			mBenchFrameCount = static_cast<float>(frameCount);
			mBenchTimeMs = elapsedMS;
			mBenchStats = stats;
		}

		inline static const char* PathFile = "bench-path.json";

	private:
		void DrawImpl() override;

		// recording
		float mTime = 1;
		CameraPath* mPath = nullptr;

		// playback
		float mPlayTime = 0;
		bool mStartPlayback = false;
		bool mStopPlayback = false;

		// benchmark result data
		float mBenchFrameCount = 0;
		float mBenchTimeMs = 0;
		Tracer::RenderStatistics mBenchStats;
	};
}
