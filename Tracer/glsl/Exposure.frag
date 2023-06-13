#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



void main()
{
	outColor = vec4(texture(convergeBuffer, inUV).xyz * exposure, 1.f);
}
