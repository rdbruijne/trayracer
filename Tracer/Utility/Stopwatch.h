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
		typedef std::chrono::steady_clock clock;
		clock::time_point mTimePoint = clock::time_point();
	};
}
