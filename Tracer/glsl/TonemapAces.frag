#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 2.2f;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



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



void main()
{
	// fetch pixel
	vec3 c = texture(convergeBuffer, inUV).xyz;
	c = c * exposure;

	// tonemapping
	c = TonemapAces(c);

	// gamma correction
	const float g = 1.f / gamma;
	c = vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));

	outColor = vec4(c, 1.f);
}
