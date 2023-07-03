// Preetham Sky Model
// Adapted from https://github.com/Tw1ddle/Sky-Shader/blob/master/src/shaders/glsl/sky.fragment

#pragma once

#include "CudaGlobals.h"
#include "CudaLinearMath.h"

// mie
#define SKY_MIE_COEFFICIENT						0.005f
#define SKY_MIE_DIRECTIONAL_G					0.80f
#define SKY_MIE_K_COEFFICIENT					make_float3(0.686f, 0.678f, 0.666f)
#define SKY_MIE_V								4.f
#define SKY_MIE_ZENITH_LENGTH					1.25e3f

// Rayleigh
#define SKY_RAYLEIGH_COEFFICIENT				2.5f
#define SKY_RAYLEIGH_ZENITH_LENGTH				8.4e3f

// atmospheric scattering
#define SKY_DEPOLARIZATION_FACTOR				0.031f
#define SKY_NUM_MOLECULES						2.545e25f
#define SKY_REFRACTIVE_INDEX					1.0003f

// wavelength for primaries
#define SKY_PRIMARY_WAVE_LENGTHS				make_float3(620e-9f, 492e-9f, 390e-9f)

// sun
//#define SKY_SUN_ANGULAR_DIAMETER_COS			0.99995541f	// cos(32 arc minutes)
//#define SKY_SUN_INTENSITY						1000.f

// Earth shadow hack
#define SKY_SUN_FALLOFF_ANGLE					(Pi / 1.95f)
#define SKY_SUN_INTENSITY_FALLOFF_STEEPNESS		1.5f

// other
#define SKY_LUMINANCE							1.f

// world info
#define SKY_UP									make_float3(0.f, 1.f, 0.f)



static __inline__ __device__
float3 TotalRayleigh(const float3& lambda)
{
	return	(make_float3(8.f * powf(Pi, 3.f) * powf(powf(SKY_REFRACTIVE_INDEX, 2.f) - 1.f, 2.f) * (6.f + 3.f * SKY_DEPOLARIZATION_FACTOR))) /
			(make_float3(3.f * SKY_NUM_MOLECULES) * pow(lambda, 4.f) * make_float3(6.f - 7.f * SKY_DEPOLARIZATION_FACTOR));
}



static __inline__ __device__
float RayleighPhase(float cosTheta)
{
	return (3.f / (16.f * Pi)) * (1.f + powf(cosTheta, 2.f));
}



static __inline__ __device__
float3 TotalMie(const float3& lambda, const float3& K, float T)
{
	const float c = 0.2f * T * 10e-18f;
	return 0.434f * c * Pi * pow((2.f * Pi) / lambda, SKY_MIE_V - 2.f) * K;
}



static __inline__ __device__
float HenyeyGreensteinPhase(float cosTheta, float g)
{
	return (1.f / (4.f * Pi)) * ((1.f - powf(g, 2.f)) / powf(1.f - 2.f * g * cosTheta + pow(g, 2.f), 1.5f));
}



static __inline__ __device__
float SunIntensity(float zenithAngleCos)
{
	return Sky->sunIntensity * max(0.f, 1.f - expf(-((SKY_SUN_FALLOFF_ANGLE - acosf(zenithAngleCos)) / SKY_SUN_INTENSITY_FALLOFF_STEEPNESS)));
}



static __inline__ __device__
float3 Tonemap(const float3& color)
{
	return (color * (1.f + color / 1000.f)) / (color + 128.f);
}



static __inline__ __device__
float3 Saturation(const float3& L0)
{
	const float3 lumaPix = make_float3(0.299f, 0.587f, 0.114f);
	const float dotLuma = dot(L0, lumaPix);
	return mix(make_float3(dotLuma, dotLuma, dotLuma), L0, 0.5f);
}



static __inline__ __device__
float3 SampleSky(const float3& sampleDir, bool drawSun)
{
	// return black when the sky is disabled
	if(Sky->skyEnabled == 0)
		return make_float3(0);

	// cache local
	const float3 sunDir = Sky->sunDir;
	const float sunAngularDiameterCos = Sky->cosSunAngularDiameter;

	// cos angles
	const float cosSunUpAngle = dot(sunDir, SKY_UP);

	// Rayleigh coefficient
	float sunfade = 1.f - __saturatef(1.f - expf(-sunDir.z / 500.f));
	const float rayleighCoefficient = SKY_RAYLEIGH_COEFFICIENT - (1.f * (1.f - sunfade));
	const float3 betaR = TotalRayleigh(SKY_PRIMARY_WAVE_LENGTHS) * rayleighCoefficient;

	// Mie coefficient
	//const float turbidity = mix(2.2f, 0.2f, max(sampleDir.y, 0.f));
	const float turbidity = Sky->turbidity;
	const float3 betaM = TotalMie(SKY_PRIMARY_WAVE_LENGTHS, SKY_MIE_K_COEFFICIENT, turbidity) * SKY_MIE_COEFFICIENT;

	// Optical length, cutoff angle at 90 to avoid singularity
	float zenithAngle = acosf(max(dot(SKY_UP, sampleDir), 0.f));
	const float denom = cos(zenithAngle) + 0.15f * pow(93.885f - ((zenithAngle * 180.f) / Pi), -1.253f);
	const float sR = SKY_RAYLEIGH_ZENITH_LENGTH / denom;
	const float sM = SKY_MIE_ZENITH_LENGTH / denom;

	// Combined extinction factor
	const float3 Fex = expf(-(betaR * sR + betaM * sM));

	// In-scattering
	const float cosTheta = dot(sampleDir, sunDir);
	const float3 betaRTheta = betaR * RayleighPhase(cosTheta * 0.5f + 0.5f);
	const float3 betaMTheta = betaM * HenyeyGreensteinPhase(cosTheta, SKY_MIE_DIRECTIONAL_G);
	const float sunE = SunIntensity(cosSunUpAngle);
	float3 Lin = pow(sunE * ((betaRTheta + betaMTheta) / (betaR + betaM)) * (1.f - Fex), 1.5f);
	Lin *= mix(make_float3(1.f),
			   pow(sunE * ((betaRTheta + betaMTheta) / (betaR + betaM)) * Fex, 0.5f),
			   __saturatef(powf(1.f - cosSunUpAngle, 5.f)));

	// Composition + solar disc
	const float sundisk = smoothstep(sunAngularDiameterCos, sunAngularDiameterCos + 2e-5f, cosTheta);
	float3 L0 = (make_float3(0.1f) * Fex) + (sunE * 19000.f * Fex * sundisk * drawSun);
	L0 = Saturation(L0);

	// Tonemapping
	const float3 color = fmaxf(make_float3(0.f), Tonemap(Lin + L0));
	sunfade *= mix(1.f, 0.9f, max(sunDir.y, 0.f));
	return pow(color, 1.f / (1.f + (sunfade)));
}
