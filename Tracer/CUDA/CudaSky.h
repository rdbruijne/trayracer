// Hosek-Wilkie Sky Model
// Adapted from ArHosekSkyModel.cpp

#pragma once

#include "CudaGlobals.h"
#include "CUDA/helper_math.h"

namespace
{
	static __device__
	float3 XYZ_to_ACES2065_1(const float3& color)
	{
		return make_float3(
			color.x *  1.0498110175f + color.y * 0.0000000000f + color.z * -0.0000974845f,
			color.x * -0.4959030231f + color.y * 1.3733130458f + color.z *  0.0982400361f,
			color.x *  0.0000000000f + color.y * 0.0000000000f + color.z *  0.9912520182f);
	}



	static __device__
	float3 ACES2065_1_to_ACEScg(const float3& color)
	{
		return make_float3(
			color.x *  1.4514393161f + color.y * -0.2365107469f + color.z * -0.2149285693f,
			color.x * -0.0765537733f + color.y *  1.1762296998f + color.z * -0.0996759265f,
			color.x *  0.0083161484f + color.y * -0.0060324498f + color.z *  0.9977163014f);
	}



	static __device__
	float3 ACES2065_1_to_sRGB(const float3& color)
	{
		return make_float3(
			color.x *  2.5216494298f + color.y * -1.1368885542f + color.z * -0.3849175932f,
			color.x * -0.2752135512f + color.y *  1.3697051510f + color.z * -0.0943924508f,
			color.x * -0.0159250101f + color.y * -0.1478063681f + color.z *  1.1638058159f);
	}




	static __device__
	float ArHosekSkyModel_GetRadianceInternal(float* config, float theta, float gamma)
	{
		const float A = config[0];
		const float B = config[1];
		const float C = config[2];
		const float D = config[3];
		const float E = config[4];
		const float F = config[5];
		const float G = config[6];
		const float H = config[7];
		const float I = config[8];

		const float cosTheta = cosf(theta);
		const float cosGamma = cosf(gamma);

		const float expM = expf(E * gamma);
		const float rayM = cosGamma * cosGamma;
		const double mieM = (1.0f + cosGamma*cosGamma) / powf((1.0f + I*I - 2.0f*I*cosGamma), 1.5f);
		const double zenith = sqrt(cosTheta);

		return (1.0f + A * expf(B / (cosTheta + 0.01f))) * (C + (D * expM) + (F * rayM) + (G * mieM) + (H * zenith));
	}



	static __device__
	float ArHosekTristimSkymodel_Radiance(SkyState* state, float theta, float gamma, int channel)
	{
		return ArHosekSkyModel_GetRadianceInternal(state->configs[channel], theta, gamma) * state->radiances[channel];
	}
}



static __device__
float3 SampleSky(float3 dir)
{
	if(dir.y < 1e-4f)
		dir.y = 1e-4f;
	dir = normalize(dir);

	const float gamma = acosf(dot(dir, skyData->sunDir));
	const float theta = acosf(dot(dir, make_float3(0, 1, 0)));

	float3 radiance;
	radiance.x = ArHosekTristimSkymodel_Radiance(skyStateX, theta, gamma, 0);
	radiance.y = ArHosekTristimSkymodel_Radiance(skyStateY, theta, gamma, 1);
	radiance.z = ArHosekTristimSkymodel_Radiance(skyStateZ, theta, gamma, 2);
	radiance = XYZ_to_ACES2065_1(radiance);

	radiance *= skyData->skyTint;
	if(skyData->enableSun && dot(dir, skyData->sunDir) >= skyData->sunSize)
		radiance += skyData->sunColor * skyData->sunTint;

	radiance = ACES2065_1_to_sRGB(radiance);
	if(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
		return make_float3(0, 0, 0);

	return radiance * .05f;
}
