#pragma once

#pragma warning(push)
#pragma warning(disable: 4061)  // enumerator 'identifier' in switch of enum 'enumeration' is not explicitly handled by a case label
#pragma warning(disable: 4365)  // 'action' : conversion from 'type_1' to 'type_2', signed/unsigned mismatch
#pragma warning(disable: 5039)  // 'function': pointer or reference to potentially throwing function passed to extern C function under -EHc. Undefined behavior may occur if this function throws an exception.
#pragma warning(disable: 6011)  // dereferencing NULL pointer <name>
#pragma warning(disable: 6387)  // <argument> may be <value>: this does not adhere to the specification for the function <function name>: Lines: x, y
#pragma warning(disable: 26451) // Arithmetic overflow: Using operator '%operator%' on a %size1% byte value and then casting the result to a %size2% byte value. Cast the value to the wider type before calling operator '%operator%' to avoid overflow

// CUDA
#include <cuda_runtime.h>

// OptiX
#include "optix7/optix.h"
#include "optix7/optix_stubs.h"
#include "optix7/optix_function_table.h"

#pragma warning(pop)
