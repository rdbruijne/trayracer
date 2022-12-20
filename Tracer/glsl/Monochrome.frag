#version 430

uniform sampler2D convergeBuffer;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;

void main()
{
	const vec3 c = texture(convergeBuffer, inUV).xyz;
	const float g = (0.299f * c.r) + (0.587f * c.g) + (0.114f * c.b);
	outColor = vec4(g, g, g, 1.f);
}
