#version 430

uniform sampler2D convergeBuffer;
uniform float gamma = 2.2f;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



void main()
{
	// fetch pixel
	vec3 c = texture(convergeBuffer, inUV).xyz;

	// gamma correction
	const float g = 1.f / gamma;
	c = vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));

	outColor = vec4(c, 1.f);
}
