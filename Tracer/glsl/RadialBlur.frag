#version 430

uniform sampler2D convergeBuffer;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;

#define T texture(convergeBuffer,.5+(p.xy*=.992))
void main()
{
	const vec3 c = texture(convergeBuffer, inUV).xyz;
	vec3 p = vec3(inUV - .5, 0);
  	vec3 o = T.xxz;
  	for (float i=0.;i<100.;i++) 
	    p.z += pow(max(0.,.5-length(T.rg)),2.)*exp(-i*.08);

	outColor = vec4(o*o+p.z,1);
}
