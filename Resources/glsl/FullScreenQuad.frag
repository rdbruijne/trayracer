#version 430

uniform sampler2D convergeBuffer;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;

void main()
{
	vec4 c = texture(convergeBuffer, inUV);
	outColor = (vec4(c.xyz, 1.0) / c.w);
}
