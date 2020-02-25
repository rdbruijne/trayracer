#include "Window.h"

// Project
#include "Utility/LinearMath.h"
#include "Utility/Utility.h"
#include "Utility/WindowsLean.h"

// GL
#include "GL/glew.h"
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
		glfwSwapInterval(0);

		// set window callbacks
		glfwSetKeyCallback(mHandle, Window::KeyCallback);
		//glfwSetCharCallback(mHandle, Window::CharCallback);
		//glfwSetCharModsCallback(mHandle, Window::CharModsCallback);
		glfwSetMouseButtonCallback(mHandle, Window::MouseButtonCallback);
		glfwSetCursorPosCallback(mHandle, Window::CursorPosCallback);
		glfwSetCursorEnterCallback(mHandle, Window::CursorEnterCallback);
		glfwSetScrollCallback(mHandle, Window::ScrollCallback);
		//glfwSetDropCallback(mHandle, Window::DropCallback);

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



	void Window::SetResolution(const int2& resolution)
	{
		// #TODO: window resizing
		assert(false);
	}



	void Window::Display(const std::vector<uint32_t>& pixels)
	{
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glClearColor(0.2f, 0.2f, 0.2f, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		// copy render target result to OpenGL buffer
		glBindTexture(GL_TEXTURE_2D, mGLTexture);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, mGLTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mResolution.x, mResolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
		glBindTexture(GL_TEXTURE_2D, 0);

		// draw a fullscreen quad
		glDisable(GL_LIGHTING);
		glColor3f(1, 1, 1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, mGLTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glDisable(GL_DEPTH_TEST);

		glViewport(0, 0, mResolution.x, mResolution.y);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.f, static_cast<float>(mResolution.x), static_cast<float>(mResolution.y), 0.f, -1.f, 1.f);

		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.f, 1.f);
			glVertex3f(0.f, 0.f, 0.f);

			glTexCoord2f(0.f, 0.f);
			glVertex3f(0.f, static_cast<float>(mResolution.y), 0.f);

			glTexCoord2f(1.f, 0.f);
			glVertex3f(static_cast<float>(mResolution.x), static_cast<float>(mResolution.y), 0.f);

			glTexCoord2f(1.f, 1.f);
			glVertex3f(static_cast<float>(mResolution.x), 0.f, 0.f);
		}
		glEnd();
	}



	void Window::SwapBuffers()
	{
		glfwSwapBuffers(mHandle);
	}



	//
	// Input
	//
	void Window::UpdateInput()
	{
		glfwPollEvents();

		mPrevInputState = mCurInputState;
		mCurInputState  = mNextInputState;
	}



	bool Window::IsKeyDown(Input::Keys key) const
	{
		if(key < Input::Keys::_LastKeyboard)
			return mCurInputState.Keyboard[static_cast<size_t>(key)];

		if(key >= Input::Keys::_FirstMouse && key <= Input::Keys::_LastMouse)
			return mCurInputState.Mouse[static_cast<size_t>(key) - static_cast<size_t>(Input::Keys::_FirstMouse)];

		assert(false);
		return false;
	}



	bool Window::WasKeyPressed(Input::Keys key) const
	{
		if(key < Input::Keys::_LastKeyboard)
		{
			const size_t keyIx = static_cast<size_t>(key);
			return mPrevInputState.Keyboard[keyIx] && !mCurInputState.Keyboard[keyIx];
		}

		if(key >= Input::Keys::_FirstMouse && key <= Input::Keys::_LastMouse)
		{
			const size_t keyIx = static_cast<size_t>(key) - static_cast<size_t>(Input::Keys::_FirstMouse);
			return mPrevInputState.Mouse[keyIx] && !mCurInputState.Mouse[keyIx];
		}

		assert(false);
		return false;
	}



	bool Window::IsCursorWithinWindow() const
	{
		return mCurInputState.MouseIsWithinWindow;
	}



	float2 Window::GetCursorPos() const
	{
		return mCurInputState.MousePos;
	}



	float2 Window::GetScroll() const
	{
		return mCurInputState.MouseScroll;
	}



	float2 Window::GetCursorDelta() const
	{
		return mCurInputState.MousePos - mPrevInputState.MousePos;
	}



	float2 Window::GetScrollDelta() const
	{
		return mCurInputState.MouseScroll - mPrevInputState.MouseScroll;
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// GLFW Input callbacks
	//--------------------------------------------------------------------------------------------------------------------------
	void Window::ErrorCallback(int error, const char* description) noexcept
	{
		printf("GLFW error %i: %s\n", error, description);
	}



	void Window::KeyCallback(GLFWwindow* handle, int key, int scancode, int action, int mods) noexcept
	{
		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		if(key < static_cast<int>(Input::Keys::_KeyboardCount))
			window->mNextInputState.Keyboard[static_cast<size_t>(key)] = (action == GLFW_PRESS || action == GLFW_REPEAT);
	}



	void Window::CharCallback(GLFWwindow* handle, unsigned int codepoint) noexcept
	{
		//Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
	}



	void Window::CharModsCallback(GLFWwindow* handle, unsigned int codepoint, int mods) noexcept
	{
		//Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
	}



	void Window::MouseButtonCallback(GLFWwindow* handle, int button, int action, int mods) noexcept
	{
		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		if(button < static_cast<int>(Input::Keys::_MouseCount))
			window->mNextInputState.Mouse[static_cast<size_t>(button)] = (action == GLFW_PRESS);
	}



	void Window::CursorPosCallback(GLFWwindow* handle, double xPos, double yPos) noexcept
	{
		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		window->mNextInputState.MousePos = make_float2(static_cast<float>(xPos), static_cast<float>(yPos));
	}



	void Window::CursorEnterCallback(GLFWwindow* handle, int entered) noexcept
	{
		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		window->mNextInputState.MouseIsWithinWindow = !entered;
	}



	void Window::ScrollCallback(GLFWwindow* handle, double xOffset, double yOffset) noexcept
	{
		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		window->mNextInputState.MouseScroll += make_float2(static_cast<float>(xOffset), static_cast<float>(yOffset));
	}



	void Window::DropCallback(GLFWwindow* handle, int count, const char** paths) noexcept
	{
		//Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
	}
}
