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
		int64_t ElapsedNS() const;
		std::string ElapsedString() const;

	private:
		std::chrono::high_resolution_clock::time_point mTimePoint = std::chrono::high_resolution_clock::time_point();
	};
}
