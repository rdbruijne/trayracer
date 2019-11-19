#pragma once

// C++
#include <chrono>
#include <string>

namespace Tracer
{
	/*!
	 * @brief Timer using std::chrone
	 */
	class Stopwatch
	{
	public:
		/*!
		 * @brief Create stopwatch
		 *
		 * The stopwatch will [reset](@ref Reset) on creation.
		 */
		Stopwatch();

		/*! Reset the stopwatch */
		void Reset();

		/*! Get the elapsed time since the last reset (in nano-seconds). */
		int64_t GetElapsedTimeNS() const;

		/*! Get the elapsed time since the last reset as a formatted string. */
		std::string GetElapsedTimeAsString() const;

	private:
		/*! Timepoint for the last reset */
		std::chrono::high_resolution_clock::time_point mTimePoint = std::chrono::high_resolution_clock::time_point();
	};
}
