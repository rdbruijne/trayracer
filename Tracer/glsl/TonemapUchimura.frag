#version 430

uniform sampler2D convergeBuffer;
uniform float exposure = 1;
uniform float gamma = 2.2f;

// passed from vertex shader
layout(location = 0) in vec2 inUV;

// result
layout(location = 0) out vec4 outColor;



vec3 TonemapUchimura(vec3 x)
{
	// Uchimura 2017, "HDR theory and practice"
	// Math: https://www.desmos.com/calculator/gslcdxvipg
	// Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
	const float P = 1.0f;  // max display brightness
	const float a = 1.0f;  // contrast
	const float m = 0.22f; // linear section start
	const float l = 0.4f;  // linear section length
	const float c = 1.33f; // black
	const float b = 0.0f;  // pedestal

	//return uchimura(x, P, a, m, l, c, b);
	const float l0 = ((P - m) * l) / a;
	const float L0 = m - m / a;
	const float L1 = m + (1.0f - m) / a;
	const float S0 = m + l0;
	const float S1 = m + a * l0;
	const float C2 = (a * P) / (P - S1);
	const float CP = -C2 / P;

	const vec3 w0 = vec3(1.0f - smoothstep(0.0f, m, x));
	const vec3 w2 = vec3(step(m + l0, x));
	const vec3 w1 = vec3(1.0f - w0 - w2);

	const vec3 T = vec3(m * pow(x / m, vec3(c)) + b);
	const vec3 S = vec3(P - (P - S1) * exp(CP * (x - S0)));
	const vec3 L = vec3(m + a * (x - m));

	return T * w0 + L * w1 + S * w2;
}



void main()
{
	// fetch pixel
	vec3 c = texture(convergeBuffer, inUV).xyz;
	c = c * exposure;

	// tonemapping
	c = TonemapUchimura(c);

	// gamma correction
	const float g = 1.f / gamma;
	c = vec3(pow(c.x, g), pow(c.y, g), pow(c.z, g));

	outColor = vec4(c, 1.f);
}
