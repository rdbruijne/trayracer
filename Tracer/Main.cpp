// Project
#include "OptixHelpers.h"
#include "Window.h"

// C++
#include <iostream>

/** Main entry point into the program.
 */
int main(int argc, char** argv)
{
	// create window
	Tracer::Window* window = new Tracer::Window("Tracer", make_int2(1280, 720));

	// init OptiX
	const bool initOptix = Tracer::OptixHelpers::Init();
	if(!initOptix)
	{
		printf("Failed to init OptiX.\n");
		return -1;
	}

	printf("Successfully initialized OptiX.\n");


	// main loop
	while (!window->WindowClosed())
	{
		// user input
		window->UpdateInput();

		// run window shaders
		window->Display();
	}

	// cleanup
	delete window;

	return 0;
}
