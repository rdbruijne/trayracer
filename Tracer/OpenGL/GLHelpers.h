#pragma once

#define GL_CHECK()	Tracer::CheckGL(__FILE__, __LINE__)

namespace Tracer
{
	void CheckGL(const char* file, int line);

	void InitGL();
	void TerminateGL();
}
