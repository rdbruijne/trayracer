#include "Window.h"

// Windows
#include "WindowsLean.h"

// GL
#include "glew/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// C++
#include <assert.h>

namespace Tracer
{
	Window::Window(const std::string& title, const int2& resolution, bool fullscreen /*= false*/) :
		mResolution(resolution)
	{
		glfwSetErrorCallback(ErrorCallback);

		// init GLFW
		if(glfwInit() != GLFW_TRUE)
			exit(EXIT_FAILURE);

		// create window
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		mHandle = glfwCreateWindow(resolution.x, resolution.y, title.c_str(), fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);
		if(!mHandle)
		{
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		// user pointer setup
		glfwSetWindowUserPointer(mHandle, this);
		glfwMakeContextCurrent(mHandle);

		// enable VSync
		glfwSwapInterval(1);

		// set window callbacks
		glfwSetKeyCallback(mHandle, Window::KeyCallback);
		glfwSetCharCallback(mHandle, Window::CharCallback);
		glfwSetCharModsCallback(mHandle, Window::CharModsCallback);
		glfwSetMouseButtonCallback(mHandle, Window::MouseButtonCallback);
		glfwSetCursorPosCallback(mHandle, Window::CursorPosCallback);
		glfwSetCursorEnterCallback(mHandle, Window::CursorEnterCallback);
		glfwSetScrollCallback(mHandle, Window::ScrollCallback);
		glfwSetDropCallback(mHandle, Window::DropCallback);

		// init GL
		const GLenum glewResult = glewInit();
		assert(glewResult == GLEW_OK);

		// viewport
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 1, 0, 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glViewport(0, 0, mResolution.x, resolution.y);

		// settings
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		// GL PBO (RGBA32F)
		glGenBuffers(1, &mGLPbo);
		assert(mGLPbo != 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mGLPbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(resolution.x * resolution.y * sizeof(float) * 4), nullptr, GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// GL texture
		glGenTextures(1, &mGLTexture);
		glBindTexture(GL_TEXTURE_2D, mGLTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);
	}



	Window::~Window()
	{
		if (mGLTexture)
			glDeleteTextures(1, &mGLTexture);

		if (mGLPbo)
			glDeleteBuffers(1, &mGLPbo);

		if (mHandle)
			glfwDestroyWindow(mHandle);

		glfwTerminate();
	}



	bool Window::IsClosed() const
	{
		return glfwWindowShouldClose(mHandle) != 0;
	}



	void Window::SetTitle(const std::string& title)
	{
		glfwSetWindowTitle(mHandle, title.c_str());
	}



	int2 Window::GetPosition() const
	{
		int2 position;
		glfwGetWindowSize(mHandle, &position.x, &position.y);
		return position;
	}



	void Window::SetPosition(const int2& position)
	{
		glfwSetWindowPos(mHandle, position.x, position.y);
	}



	int2 Window::GetResolution() const
	{
		int2 resolution;
		glfwGetWindowSize(mHandle, &resolution.x, &resolution.y);
		return resolution;
	}



	void Window::Display()
	{
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glClearColor(0.2f, 0.2f, 0.2f, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		glfwSwapBuffers(mHandle);
	}



	void Window::UpdateInput()
	{
		glfwPollEvents();
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// GLFW Input callbacks
	//--------------------------------------------------------------------------------------------------------------------------
	void Window::ErrorCallback(int error, const char* description)
	{
		printf("GLFW error %i: %s\n", error, description);
	}



	void Window::KeyCallback(GLFWwindow* handle, int key, int scancode, int action, int mods)
	{
	}



	void Window::CharCallback(GLFWwindow* handle, unsigned int codepoint)
	{
	}



	void Window::CharModsCallback(GLFWwindow* handle, unsigned int codepoint, int mods)
	{
	}



	void Window::MouseButtonCallback(GLFWwindow* handle, int button, int action, int mods)
	{
	}



	void Window::CursorPosCallback(GLFWwindow* handle, double xPos, double yPos)
	{
	}



	void Window::CursorEnterCallback(GLFWwindow* handle, int entered)
	{
	}



	void Window::ScrollCallback(GLFWwindow* handle, double xOffset, double yOffset)
	{
	}



	void Window::DropCallback(GLFWwindow* handle, int count, const char** paths)
	{
	}
}
