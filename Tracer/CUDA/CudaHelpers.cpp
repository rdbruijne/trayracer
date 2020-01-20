#include "CUDA/CudaHelpers.h"

#pragma warning(push)
#pragma warning(disable: 26812) // The enum type %type% is unscoped. Prefer 'enum class' over 'enum'
std::string Tracer::ToString(CUresult cuResult)
{
	switch(cuResult)
	{
		case CUDA_SUCCESS: // 0
			return "CUDA_SUCCESS";

		case CUDA_ERROR_INVALID_VALUE: // 1
			return "CUDA_ERROR_INVALID_VALUE";

		case CUDA_ERROR_OUT_OF_MEMORY: // 2
			return "CUDA_ERROR_OUT_OF_MEMORY";

		case CUDA_ERROR_NOT_INITIALIZED: // 3
			return "CUDA_ERROR_NOT_INITIALIZED";

		case CUDA_ERROR_DEINITIALIZED: // 4
			return "CUDA_ERROR_DEINITIALIZED";

		case CUDA_ERROR_PROFILER_DISABLED: // 5
			return "CUDA_ERROR_PROFILER_DISABLED";

		case CUDA_ERROR_PROFILER_NOT_INITIALIZED: // 6
			return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

		case CUDA_ERROR_PROFILER_ALREADY_STARTED: // 7
			return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

		case CUDA_ERROR_PROFILER_ALREADY_STOPPED: // 8
			return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

		case CUDA_ERROR_NO_DEVICE: // 100
			return "CUDA_ERROR_NO_DEVICE";

		case CUDA_ERROR_INVALID_DEVICE: // 101
			return "CUDA_ERROR_INVALID_DEVICE";

		case CUDA_ERROR_INVALID_IMAGE: // 200
			return "CUDA_ERROR_INVALID_IMAGE";

		case CUDA_ERROR_INVALID_CONTEXT: // 201
			return "CUDA_ERROR_INVALID_CONTEXT";

		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: // 202
			return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

		case CUDA_ERROR_MAP_FAILED: // 205
			return "CUDA_ERROR_MAP_FAILED";

		case CUDA_ERROR_UNMAP_FAILED: // 206
			return "CUDA_ERROR_UNMAP_FAILED";

		case CUDA_ERROR_ARRAY_IS_MAPPED: // 207
			return "CUDA_ERROR_ARRAY_IS_MAPPED";

		case CUDA_ERROR_ALREADY_MAPPED: // 208
			return "CUDA_ERROR_ALREADY_MAPPED";

		case CUDA_ERROR_NO_BINARY_FOR_GPU: // 209
			return "CUDA_ERROR_NO_BINARY_FOR_GPU";

		case CUDA_ERROR_ALREADY_ACQUIRED: // 210
			return "CUDA_ERROR_ALREADY_ACQUIRED";

		case CUDA_ERROR_NOT_MAPPED: // 211
			return "CUDA_ERROR_NOT_MAPPED";

		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: // 212
			return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

		case CUDA_ERROR_NOT_MAPPED_AS_POINTER: // 213
			return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

		case CUDA_ERROR_ECC_UNCORRECTABLE: // 214
			return "CUDA_ERROR_ECC_UNCORRECTABLE";

		case CUDA_ERROR_UNSUPPORTED_LIMIT: // 215
			return "CUDA_ERROR_UNSUPPORTED_LIMIT";

		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: // 216
			return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

		case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: // 217
			return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";

		case CUDA_ERROR_INVALID_PTX: // 218
			return "CUDA_ERROR_INVALID_PTX";

		case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: // 219
			return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";

		case CUDA_ERROR_NVLINK_UNCORRECTABLE: // 220
			return "CUDA_ERROR_NVLINK_UNCORRECTABLE";

		case CUDA_ERROR_JIT_COMPILER_NOT_FOUND: // 221
			return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";

		case CUDA_ERROR_INVALID_SOURCE: // 300
			return "CUDA_ERROR_INVALID_SOURCE";

		case CUDA_ERROR_FILE_NOT_FOUND: // 301
			return "CUDA_ERROR_FILE_NOT_FOUND";

		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: // 302
			return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: // 303
			return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

		case CUDA_ERROR_OPERATING_SYSTEM: // 304
			return "CUDA_ERROR_OPERATING_SYSTEM";

		case CUDA_ERROR_INVALID_HANDLE: // 400
			return "CUDA_ERROR_INVALID_HANDLE";

		case CUDA_ERROR_ILLEGAL_STATE: // 401
			return "CUDA_ERROR_ILLEGAL_STATE";

		case CUDA_ERROR_NOT_FOUND: // 500
			return "CUDA_ERROR_NOT_FOUND";

		case CUDA_ERROR_NOT_READY: // 600
			return "CUDA_ERROR_NOT_READY";

		case CUDA_ERROR_ILLEGAL_ADDRESS: // 700
			return "CUDA_ERROR_ILLEGAL_ADDRESS";

		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: // 701
			return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

		case CUDA_ERROR_LAUNCH_TIMEOUT: // 702
			return "CUDA_ERROR_LAUNCH_TIMEOUT";

		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: // 703
			return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: // 704
			return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: // 705
			return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: // 708
			return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

		case CUDA_ERROR_CONTEXT_IS_DESTROYED: // 709
			return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

		case CUDA_ERROR_ASSERT: // 710
			return "CUDA_ERROR_ASSERT";

		case CUDA_ERROR_TOO_MANY_PEERS: // 711
			return "CUDA_ERROR_TOO_MANY_PEERS";

		case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: // 712
			return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

		case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: // 713
			return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

		case CUDA_ERROR_HARDWARE_STACK_ERROR: // 714
			return "CUDA_ERROR_HARDWARE_STACK_ERROR";

		case CUDA_ERROR_ILLEGAL_INSTRUCTION: // 715
			return "CUDA_ERROR_ILLEGAL_INSTRUCTION";

		case CUDA_ERROR_MISALIGNED_ADDRESS: // 716
			return "CUDA_ERROR_MISALIGNED_ADDRESS";

		case CUDA_ERROR_INVALID_ADDRESS_SPACE: // 717
			return "CUDA_ERROR_INVALID_ADDRESS_SPACE";

		case CUDA_ERROR_INVALID_PC: // 718
			return "CUDA_ERROR_INVALID_PC";

		case CUDA_ERROR_LAUNCH_FAILED: // 719
			return "CUDA_ERROR_LAUNCH_FAILED";

		case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: // 720
			return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";

		case CUDA_ERROR_NOT_PERMITTED: // 800
			return "CUDA_ERROR_NOT_PERMITTED";

		case CUDA_ERROR_NOT_SUPPORTED: // 801
			return "CUDA_ERROR_NOT_SUPPORTED";

		case CUDA_ERROR_SYSTEM_NOT_READY: // 802
			return "CUDA_ERROR_SYSTEM_NOT_READY";

		case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: // 803
			return "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";

		case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: // 804
			return "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE";

		case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: // 900
			return "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";

		case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: // 901
			return "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";

		case CUDA_ERROR_STREAM_CAPTURE_MERGE: // 902
			return "CUDA_ERROR_STREAM_CAPTURE_MERGE";

		case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: // 903
			return "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";

		case CUDA_ERROR_STREAM_CAPTURE_UNJOINED: // 904
			return "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";

		case CUDA_ERROR_STREAM_CAPTURE_ISOLATION: // 905
			return "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";

		case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: // 906
			return "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";

		case CUDA_ERROR_CAPTURED_EVENT: // 907
			return "CUDA_ERROR_CAPTURED_EVENT";

		case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: // 908
			return "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";

		case CUDA_ERROR_UNKNOWN: // 999
			return "CUDA_ERROR_UNKNOWN";

		default:
			return std::to_string(cuResult);
	}
}
#pragma warning(pop)
