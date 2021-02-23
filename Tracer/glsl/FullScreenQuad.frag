#version 430

uniform sampler2D convergeBuffer;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



// main shader program
void main()
{
	outColor = vec4(texture(convergeBuffer, inUV).xyz, 1.f);
}
