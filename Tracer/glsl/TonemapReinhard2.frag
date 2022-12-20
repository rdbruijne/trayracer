#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 2.2f;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



vec3 TonemapReinhard2(vec3 x)
{
  const float L_white = 4.0f;
  return (x * (1.0f + x / (L_white * L_white))) / (1.0f + x);
}



void main()
{
	// fetch pixel
	vec3 c = texture(convergeBuffer, inUV).xyz;
	c = c * exposure;

	// tonemapping
	c = TonemapReinhard2(c);

	// gamma correction
	const float g = 1.f / gamma;
	c = vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));

	outColor = vec4(c, 1.f);
}
