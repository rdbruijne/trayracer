#version 430
layout(location = 0) out vec2 outUV;

void main()
{
	// draw single triangle which extends beyond the screen: [-1, -1], [-1, 3], [3, -1]
	const float x = float(gl_VertexID >> 1);
	const float y = float(gl_VertexID & 1);
	outUV = vec2(x * 2.0, y * 2.0);
	gl_Position = vec4(vec2(x * 4.0 - 1.0, y * 4.0 - 1.0), 0, 1.0);
}
