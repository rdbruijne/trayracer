#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 2.2f;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



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



void main()
{
	// fetch pixel
	vec3 c = texture(convergeBuffer, inUV).xyz;
	c = c * exposure;

	// tonemapping
	c = TonemapUncharted2(c);

	// gamma correction
	const float g = 1.f / gamma;
	c = vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));

	outColor = vec4(c, 1.f);
}
