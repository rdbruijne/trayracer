#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 1;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;

void main()
{
	vec4 c = texture(convergeBuffer, inUV);
	c.xyz /= c.w;
	c.x = pow(c.x, 1.f / gamma);
	c.y = pow(c.y, 1.f / gamma);
	c.z = pow(c.z, 1.f / gamma);
	outColor = vec4(c.xyz, 1.0) * exposure;
}
