#version 430

uniform sampler2D convergeBuffer;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;

vec3 LessThan(vec3 f, float value)
{
	return vec3(
		(f.x < value) ? 1.0f : 0.0f,
		(f.y < value) ? 1.0f : 0.0f,
		(f.z < value) ? 1.0f : 0.0f);
}

void main()
{
	const vec3 rgb = clamp(texture(convergeBuffer, inUV).xyz, 0.f, 1.f);
	const vec3 srgb = mix(
		pow(rgb, vec3(1.0f / 2.4f)) * 1.055f - 0.055f,
		rgb * 12.92f,
		LessThan(rgb, 0.0031308f));
	outColor = vec4(srgb, 1.f);
}
