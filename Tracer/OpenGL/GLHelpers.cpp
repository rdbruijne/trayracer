#include "GLHelpers.h"

// Project
#include "Utility/Utility.h"
#include "Utility/Logger.h"

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



		void GlfwErrorCallback(int error, const char* description) noexcept
		{
			Logger::Error("GLFW error %i: %s", error, description);
		}
	}



	void CheckGL(const char* file, int line)
	{
		const GLenum error = glGetError();
		if(error != GL_NO_ERROR)
			FatalError("GL error in \"%s\" @ %d: %s (%#x)", file, line, GlErrorToString(error).c_str(), error);
	}



	void InitGL()
	{
		glfwSetErrorCallback(GlfwErrorCallback);

		if(glfwInit() != GLFW_TRUE)
			FatalError("Failed to init glfw");
	}



	void TerminateGL()
	{
		glfwTerminate();
	}



	const char* GLRenderer()
	{
		return reinterpret_cast<const char*>(glGetString(GL_RENDERER));
	}



	const char* GLShadingLanguageVersion()
	{
		return reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION));
	}



	const char* GLVendor()
	{
		return reinterpret_cast<const char*>(glGetString(GL_VENDOR));
	}



	const char* GLVersion()
	{
		return reinterpret_cast<const char*>(glGetString(GL_VERSION));
	}
}
