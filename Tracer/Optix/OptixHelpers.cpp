#include "Optix/OptixHelpers.h"

// Project
#include "CUDA/CudaHelpers.h"
#include "Utility/Utility.h"

// OptiX
#include "optix7/optix_function_table_definition.h"

// C++
#include <cassert>
#include <iostream>

namespace Tracer
{
	bool InitOptix()
	{
		// check for available devices
		int numDevices = 0;
		CUDA_CHECK(cudaGetDeviceCount(&numDevices));
		if(numDevices == 0)
		{
			printf("No CUDA capable devices found.\n");
			return false;
		}
		printf("Found %i CUDA capable devices.\n", numDevices);

		// init OptiX
		const OptixResult res = optixInit();
		if(res != OPTIX_SUCCESS)
		{
			printf("Failed to init OptiX: %s\n", ToString(res).c_str());
			return false;
		}

		return true;
	}



	void OptixCheck(OptixResult res, const char* file, int line)
	{
		assert(res == OPTIX_SUCCESS);
		if(res != OPTIX_SUCCESS)
		{
			const std::string errorMessage = format("OptiX error at \"%s\" @ %i: %s", file, line, ToString(res).c_str());
			throw std::runtime_error(errorMessage);
		}
	}



	std::string ToString(OptixResult optixResult)
	{
		switch(optixResult)
		{
		case OPTIX_SUCCESS:
			return "OPTIX_SUCCESS";

		case OPTIX_ERROR_INVALID_VALUE:
			return "OPTIX_ERROR_INVALID_VALUE";

		case OPTIX_ERROR_HOST_OUT_OF_MEMORY:
			return "OPTIX_ERROR_HOST_OUT_OF_MEMORY";

		case OPTIX_ERROR_INVALID_OPERATION:
			return "OPTIX_ERROR_INVALID_OPERATION";

		case OPTIX_ERROR_FILE_IO_ERROR:
			return "OPTIX_ERROR_FILE_IO_ERROR";

		case OPTIX_ERROR_INVALID_FILE_FORMAT:
			return "OPTIX_ERROR_INVALID_FILE_FORMAT";

		case OPTIX_ERROR_DISK_CACHE_INVALID_PATH:
			return "OPTIX_ERROR_DISK_CACHE_INVALID_PATH";

		case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR:
			return "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR";

		case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR:
			return "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR";

		case OPTIX_ERROR_DISK_CACHE_INVALID_DATA:
			return "OPTIX_ERROR_DISK_CACHE_INVALID_DATA";

		case OPTIX_ERROR_LAUNCH_FAILURE:
			return "OPTIX_ERROR_LAUNCH_FAILURE";

		case OPTIX_ERROR_INVALID_DEVICE_CONTEXT:
			return "OPTIX_ERROR_INVALID_DEVICE_CONTEXT";

		case OPTIX_ERROR_CUDA_NOT_INITIALIZED:
			return "OPTIX_ERROR_CUDA_NOT_INITIALIZED";

		case OPTIX_ERROR_INVALID_PTX:
			return "OPTIX_ERROR_INVALID_PTX";

		case OPTIX_ERROR_INVALID_LAUNCH_PARAMETER:
			return "OPTIX_ERROR_INVALID_LAUNCH_PARAMETER";

		case OPTIX_ERROR_INVALID_PAYLOAD_ACCESS:
			return "OPTIX_ERROR_INVALID_PAYLOAD_ACCESS";

		case OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS:
			return "OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS";

		case OPTIX_ERROR_INVALID_FUNCTION_USE:
			return "OPTIX_ERROR_INVALID_FUNCTION_USE";

		case OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS:
			return "OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS";

		case OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY:
			return "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY";

		case OPTIX_ERROR_PIPELINE_LINK_ERROR:
			return "OPTIX_ERROR_PIPELINE_LINK_ERROR";

		case OPTIX_ERROR_INTERNAL_COMPILER_ERROR:
			return "OPTIX_ERROR_INTERNAL_COMPILER_ERROR";

		case OPTIX_ERROR_DENOISER_MODEL_NOT_SET:
			return "OPTIX_ERROR_DENOISER_MODEL_NOT_SET";

		case OPTIX_ERROR_DENOISER_NOT_INITIALIZED:
			return "OPTIX_ERROR_DENOISER_NOT_INITIALIZED";

		case OPTIX_ERROR_ACCEL_NOT_COMPATIBLE:
			return "OPTIX_ERROR_ACCEL_NOT_COMPATIBLE";

		case OPTIX_ERROR_NOT_SUPPORTED:
			return "OPTIX_ERROR_NOT_SUPPORTED";

		case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
			return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";

		case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
			return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";

		case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS:
			return "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";

		case OPTIX_ERROR_LIBRARY_NOT_FOUND:
			return "OPTIX_ERROR_LIBRARY_NOT_FOUND";

		case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
			return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";

		case OPTIX_ERROR_CUDA_ERROR:
			return "OPTIX_ERROR_CUDA_ERROR";

		case OPTIX_ERROR_INTERNAL_ERROR:
			return "OPTIX_ERROR_INTERNAL_ERROR";

		case OPTIX_ERROR_UNKNOWN:
			return "OPTIX_ERROR_UNKNOWN";

		default:
			return "Unknown OptiX error code!";
		}
	}
}
