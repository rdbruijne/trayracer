#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 1;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



// gamma correction
vec3 GammaCorrect(vec3 c)
{
	float g = 1.f / gamma;
	return vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));
}



// tonemapping
vec3 Tonemap(vec3 c)
{
	return clamp(c, 0, 1);
}



// main shader program
void main()
{
	vec3 c = texture(convergeBuffer, inUV).xyz;
	c = Tonemap(c);
	c *= exposure;
	c = GammaCorrect(c);
	outColor = c;
}
