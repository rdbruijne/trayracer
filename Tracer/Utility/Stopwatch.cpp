#include "Stopwatch.h"

// Project
#include "Utility.h"

namespace Tracer
{
	Stopwatch::Stopwatch()
	{
		Reset();
	}



	void Stopwatch::Reset()
	{
		mTimePoint = std::chrono::high_resolution_clock::now();
	}



	int64_t Stopwatch::GetElapsedTimeNS() const
	{
		const std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
		const int64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t - mTimePoint).count();
		return elapsed;
	}



	std::string Stopwatch::GetElapsedTimeAsString() const
	{
		const int64_t t = GetElapsedTimeNS();
		if(t < 1'000)
			return format("%d ns", t);
		if(t < 1'000'000)
			return format("%d.%01d us", t / 1'000, (t / 100) % 10);
		if(t < 1'000'000'000)
			return format("%d.%01d ms", t / 1'000'000, (t / 100'000) % 10);
		if(t < 60'000'000'000)
			return format("%d.%01d s", t / 1'000'000'000, (t / 100'000'000) % 10);

		const int64_t t2 = t / 1'000'000'000;
		return format("%d:%d", t2 / 60, t2 % 60);
	}
}
