#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 1;
uniform int tonemapMethod = 0;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



//------------------------------------------------------------------------------------------------------------------------------
// gamma correction
//------------------------------------------------------------------------------------------------------------------------------
vec3 GammaCorrect(vec3 c)
{
	float g = 1.f / gamma;
	return vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));
}



//------------------------------------------------------------------------------------------------------------------------------
// tonemapping adapted from dmnsgn
// https://github.com/dmnsgn/glsl-tone-map
//------------------------------------------------------------------------------------------------------------------------------

// Copyright (C) 2019 the internet and Damien Seguin
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
// OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

vec3 TonemapAces(vec3 x)
{
	// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
	const float a = 2.51f;
	const float b = 0.03f;
	const float c = 2.43f;
	const float d = 0.59f;
	const float e = 0.14f;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}



vec3 TonemapFilmic(vec3 x)
{
	// Filmic Tonemapping Operators http://filmicworlds.com/blog/filmic-tonemapping-operators/
	const vec3 x2 = max(vec3(0.0f), x - 0.004f);
	const vec3 result = (x2 * (6.2f * x2 + 0.5f)) / (x2 * (6.2f * x2 + 1.7f) + 0.06f);
	return pow(result, vec3(2.2f));
}



vec3 TonemapLottes(vec3 x)
{
	// Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines"
	const vec3 a      = vec3(1.6f);
	const vec3 d      = vec3(0.977f);
	const vec3 hdrMax = vec3(8.0f);
	const vec3 midIn  = vec3(0.18f);
	const vec3 midOut = vec3(0.267f);

	const vec3 b =
		(-pow(midIn, a) + pow(hdrMax, a) * midOut) /
		((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
	const vec3 c =
		(pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
		((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

	return pow(x, a) / (pow(x, a * d) * b + c);
}



vec3 TonemapReinhard(vec3 x)
{
  return x / (1.0f + x);
}




vec3 TonemapReinhard2(vec3 x)
{
  const float L_white = 4.0f;
  return (x * (1.0f + x / (L_white * L_white))) / (1.0f + x);
}



vec3 TonemapUchimura(vec3 x)
{
	// Uchimura 2017, "HDR theory and practice"
	// Math: https://www.desmos.com/calculator/gslcdxvipg
	// Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
	const float P = 1.0f;  // max display brightness
	const float a = 1.0f;  // contrast
	const float m = 0.22f; // linear section start
	const float l = 0.4f;  // linear section length
	const float c = 1.33f; // black
	const float b = 0.0f;  // pedestal

	//return uchimura(x, P, a, m, l, c, b);
	const float l0 = ((P - m) * l) / a;
	const float L0 = m - m / a;
	const float L1 = m + (1.0f - m) / a;
	const float S0 = m + l0;
	const float S1 = m + a * l0;
	const float C2 = (a * P) / (P - S1);
	const float CP = -C2 / P;

	const vec3 w0 = vec3(1.0f - smoothstep(0.0f, m, x));
	const vec3 w2 = vec3(step(m + l0, x));
	const vec3 w1 = vec3(1.0f - w0 - w2);

	const vec3 T = vec3(m * pow(x / m, vec3(c)) + b);
	const vec3 S = vec3(P - (P - S1) * exp(CP * (x - S0)));
	const vec3 L = vec3(m + a * (x - m));

	return T * w0 + L * w1 + S * w2;
}



vec3 TonemapUncharted2Partial(vec3 x)
{
	const float A = 0.15f;
	const float B = 0.50f;
	const float C = 0.10f;
	const float D = 0.20f;
	const float E = 0.02f;
	const float F = 0.30f;
	const float W = 11.2f;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}



vec3 TonemapUncharted2(vec3 x)
{
	const float W = 11.2f;
	const float exposureBias = 2.0f;
	vec3 curr = TonemapUncharted2Partial(exposureBias * x);
	vec3 whiteScale = 1.0f / TonemapUncharted2Partial(vec3(W));
	return curr * whiteScale;
}



vec3 Tonemap(vec3 x)
{
	// see Window::ShaderProperties::TonemapMethod
	switch(tonemapMethod)
	{
	default:
	case 0:
		return TonemapAces(x);

	case 1:
		return TonemapFilmic(x);

	case 2:
		return TonemapLottes(x);

	case 3:
		return TonemapReinhard(x);

	case 4:
		return TonemapReinhard2(x);

	case 5:
		return TonemapUchimura(x);

	case 6:
		return TonemapUncharted2(x);
	}
}



//------------------------------------------------------------------------------------------------------------------------------
// main shader program
//------------------------------------------------------------------------------------------------------------------------------
void main()
{
	vec3 c = texture(convergeBuffer, inUV).xyz;
	c = c * exposure;
	c = Tonemap(c);
	c = GammaCorrect(c);
	outColor = vec4(c, 1.f);
}
