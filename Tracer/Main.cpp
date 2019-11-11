// Project
#include "OptixHelpers.h"

// C++
#include <iostream>

int main(int argc, char** argv)
{
	const bool initOptix = Tracer::OptixHelpers::Init();
	if(!initOptix)
	{
		printf("Failed to init OptiX.\n");
		return -1;
	}

	printf("Successfully initialized OptiX.\n");
	return 0;
}
