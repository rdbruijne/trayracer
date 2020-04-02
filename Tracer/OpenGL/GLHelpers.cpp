#include "GLHelpers.h"

// Project
#include "Utility/Utility.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// C++
#include <cassert>
#include <stdexcept>

namespace Tracer
{
	namespace
	{
		std::string GlErrorToString(GLenum error)
		{
			switch(error)
			{
			case 0x500:
				return "Invalid enum";

			case 0x501:
				return "Invalid value";

			case 0x502:
				return "Invalid operation";

			case 0x0503:
				return "Stack overflow";

			case 0x0504:
				return "Stack underflow";

			case 0x0505:
				return "Out of memory";

			case 0x506:
				return "Invalid framebuffer operation";

			case 0x507:
				return "GL context lost";

			default:
				return "Unknown error";
			}
		}
	}



	void CheckGL(const char* file, int line)
	{
		GLenum error = glGetError();
		assert(error == GL_NO_ERROR);
		if(error != GL_NO_ERROR)
		{
			const std::string errorMessage = format("GL error in \"%s\" @ %d: %s (%#x)", file, line, GlErrorToString(error).c_str(), error);
			throw std::runtime_error(errorMessage);
		}
	}
}
