#version 430

uniform sampler2D convergeBuffer;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;

void main()
{
	const vec3 c = texture(convergeBuffer, inUV).xyz;
	outColor = vec4(1.f - clamp(c.x, 0, 1), 1.f - clamp(c.y, 0, 1), 1.f - clamp(c.z, 0, 1), 1.f);
}
