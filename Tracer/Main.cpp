// Project
#include "OptixHelpers.h"
#include "Renderer.h"
#include "Stopwatch.h"
#include "Window.h"

// C++
#include <iostream>

/*!
 * @brief Main entry point into the program.
 */
int main(int argc, char** argv)
{
	// init OptiX
	const bool initOptix = Tracer::OptixHelpers::Init();
	if(!initOptix)
	{
		printf("Failed to init OptiX.\n");
		return -1;
	}
	printf("Successfully initialized OptiX.\n");

	const int2 renderResolution = make_int2(1280, 720);

	// create renderer
	Tracer::Renderer* renderer = new Tracer::Renderer(renderResolution);

	// create window
	Tracer::Window* window = new Tracer::Window("Tracer", renderResolution);

	// timer
	Tracer::Stopwatch stopwatch;
	int64_t elapsedNs = 0;

	// main loop
	while (!window->IsClosed())
	{
		// user input
		window->UpdateInput();

		// run OptiX
		renderer->RenderFrame();

		std::vector<uint32_t> pixels;
		renderer->DownloadPixels(pixels);

		// run window shaders
		window->Display(pixels);

		// update the title bar
		window->SetTitle(Tracer::format("Tracer - %.1f ms - %.1f FPS", static_cast<double>(elapsedNs) * 1e-3, 1e6f / elapsedNs));

		// update timer
		elapsedNs = stopwatch.GetElapsedTimeNS();
		stopwatch.Reset();
	}

	// cleanup
	delete window;

	return 0;
}
