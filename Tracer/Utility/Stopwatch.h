#pragma once

// C++
#include <chrono>
#include <string>

namespace Tracer
{
	class Stopwatch
	{
	public:
		Stopwatch();

		void Reset();

		float Elapsed() const { return static_cast<float>(ElapsedNs()) * 1e-9f; }
		float ElapsedMs() const { return static_cast<float>(ElapsedNs()) * 1e-6f; }
		float ElapsedUs() const { return static_cast<float>(ElapsedNs()) * 1e-3f; }
		int64_t ElapsedNs() const;

		std::string ElapsedString() const;

	private:
		typedef std::chrono::steady_clock clock;
		clock::time_point mTimePoint = clock::time_point();
	};
}
