#pragma once

// CUDA
#include <vector_types.h>

// OptiX
#include "Optix7.h"

// C++
#include <stdint.h>

/*!
 * @brief Program launch parameters.
 */
struct alignas(16) LaunchParams
{
	/*! Camera position. */
	float3 cameraPos;
	/*! Width of the render target. */
	int32_t resolutionX;

	/*! Camera's horizontal (side) axis. */
	float3 cameraSide;
	/*! Height of the render target. */
	int32_t resolutionY;

	/*! Camera's vertical (up) axis.*/
	float3 cameraUp;
	/*! Frame index. */
	int32_t frameID;

	/*! Camera's forward axis. */
	float3 cameraForward;
	/*! Dummy value for data alignment. */
	int32_t dummy;

	/*! OptiX root object to traverse. */
	OptixTraversableHandle sceneRoot;
	/*! Render target color buffer. */
	uint32_t* colorBuffer;
};



/*!
 * @brief Ray type identifiers
 */
enum RayTypes
{
	/*! Default ray type. */
	RAY_TYPE_SURFACE = 0,

	/*! Number of ray types. */
	RAY_TYPE_COUNT
};

