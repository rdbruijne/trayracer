#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 2.2f;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



vec3 TonemapFilmic(vec3 x)
{
	// Filmic Tonemapping Operators http://filmicworlds.com/blog/filmic-tonemapping-operators/
	const vec3 x2 = max(vec3(0.0f), x - 0.004f);
	const vec3 result = (x2 * (6.2f * x2 + 0.5f)) / (x2 * (6.2f * x2 + 1.7f) + 0.06f);
	return pow(result, vec3(2.2f));
}



void main()
{
	// fetch pixel
	vec3 c = texture(convergeBuffer, inUV).xyz;
	c = c * exposure;

	// tonemapping
	c = TonemapFilmic(c);

	// gamma correction
	const float g = 1.f / gamma;
	c = vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));

	outColor = vec4(c, 1.f);
}
