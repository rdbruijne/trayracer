#include "Window.h"

// Project
#include "OpenGL/GLTexture.h"
#include "OpenGL/Shader.h"
#include "Utility/LinearMath.h"
#include "Utility/Logger.h"
#include "Utility/Utility.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// ImGUI
#include "imgui/imgui.h"

// C++
#include <assert.h>
#include <stdexcept>

// Windows
#ifdef APIENTRY
#undef APIENTRY
#endif
#include <Windows.h>

namespace Tracer
{
	Window::Window(const std::string& title, const int2& resolution, bool fullscreen) :
		mResolution(resolution)
	{
		glfwSetErrorCallback(ErrorCallback);

		// init GLFW
		if(glfwInit() != GLFW_TRUE)
			throw std::runtime_error("Failed to init glfw");

		// create window
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		mHandle = glfwCreateWindow(resolution.x, resolution.y, title.c_str(), fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

		if(!mHandle)
		{
			glfwTerminate();
			throw std::runtime_error("Failed to create glfw window");
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
		if(glewResult != GLEW_OK)
			throw std::runtime_error("Failed to init glew");

		// dpi fix
		SetResolution(resolution);

		// settings
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		// init shader
		mShader = new Shader("glsl/FullScreenQuad.vert", "glsl/FullScreenQuad.frag");

		// GL texture
		mRenderTexture = new GLTexture(mResolution, GLTexture::Types::Float4);
	}



	Window::~Window()
	{
		delete mShader;
		delete mRenderTexture;

		if (mHandle)
			glfwDestroyWindow(mHandle);

		glfwTerminate();
	}



	void Window::Close()
	{
		glfwSetWindowShouldClose(mHandle, 1);
	}



	bool Window::IsClosed() const
	{
		return glfwWindowShouldClose(mHandle) != 0;
	}



	void Window::SetTitle(const std::string& title)
	{
		glfwSetWindowTitle(mHandle, title.c_str());
	}



	int2 Window::Position() const
	{
		int2 position;
		glfwGetWindowSize(mHandle, &position.x, &position.y);
		return position;
	}



	void Window::SetPosition(const int2& position)
	{
		glfwSetWindowPos(mHandle, position.x, position.y);
	}



	int2 Window::Resolution() const
	{
		int2 resolution;
		glfwGetWindowSize(mHandle, &resolution.x, &resolution.y);

		const float dpiScale = MonitorDPI(CurrentMonitor());
		return make_int2(static_cast<int>(resolution.x / dpiScale), static_cast<int>(resolution.y / dpiScale));
	}



	void Window::SetResolution(const int2& resolution)
	{
		if(!mRenderTexture || mRenderTexture->Resolution() != resolution)
		{
			delete mRenderTexture;
			mRenderTexture = new GLTexture(resolution, GLTexture::Types::Float4);
		}

		const float dpiScale = MonitorDPI(CurrentMonitor());
		const int2 dpiRes = make_int2(static_cast<int>(resolution.x * dpiScale), static_cast<int>(resolution.y * dpiScale));
		glfwSetWindowSize(mHandle, dpiRes.x, dpiRes.y);

		// viewport
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 1, 0, 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glViewport(0, 0, dpiRes.x, dpiRes.y);
	}



	GLTexture* Window::RenderTexture()
	{
		return mRenderTexture;
	}



	const GLTexture* Window::RenderTexture() const
	{
		return mRenderTexture;
	}



	void Window::Display()
	{
		// DPI fix
		SetResolution(Resolution());

		glEnable(GL_TEXTURE_2D);
		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);

		mShader->Bind();
		mShader->Set(0, "convergeBuffer", mRenderTexture);
		mShader->Set("exposure", mShaderProperties.exposure);
		mShader->Set("gamma",  mShaderProperties.gamma);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		mShader->Unbind();

		glFinish();
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
		if(key < Input::KeyData::LastKeyboard)
			return mCurInputState.Keyboard[static_cast<size_t>(key)];

		if(key >= Input::KeyData::FirstMouse && key <= Input::KeyData::LastMouse)
			return mCurInputState.Mouse[static_cast<size_t>(key) - static_cast<size_t>(Input::KeyData::FirstMouse)];

		if(key >= Input::KeyData::FirstSpecial && key <= Input::KeyData::LastSpecial)
			return false;

		assert(false);
		return false;
	}



	bool Window::WasKeyPressed(Input::Keys key) const
	{
		if(key < Input::KeyData::LastKeyboard)
		{
			const size_t keyIx = static_cast<size_t>(key);
			return mPrevInputState.Keyboard[keyIx] && !mCurInputState.Keyboard[keyIx];
		}

		if(key >= Input::KeyData::FirstMouse && key <= Input::KeyData::LastMouse)
		{
			const size_t keyIx = static_cast<size_t>(key) - static_cast<size_t>(Input::KeyData::FirstMouse);
			return mPrevInputState.Mouse[keyIx] && !mCurInputState.Mouse[keyIx];
		}

		if(key >= Input::KeyData::FirstSpecial && key <= Input::KeyData::LastSpecial)
			return false;

		assert(false);
		return false;
	}



	bool Window::IsCursorWithinWindow() const
	{
		return mCurInputState.MouseIsWithinWindow;
	}



	float2 Window::CursorPos() const
	{
		return mCurInputState.MousePos / MonitorDPI(CurrentMonitor());
	}



	float2 Window::Scroll() const
	{
		return mCurInputState.MouseScroll;
	}



	float2 Window::CursorDelta() const
	{
		return mCurInputState.MousePos - mPrevInputState.MousePos;
	}



	float2 Window::ScrollDelta() const
	{
		return mCurInputState.MouseScroll - mPrevInputState.MouseScroll;
	}



	int Window::CurrentMonitor() const
	{
		int count = 0;
		GLFWmonitor** monitors = glfwGetMonitors(&count);
		const int2 windowPos = Position();

		for(int i = 0; i < count; i++)
		{
			int xPos = 0;
			int yPos = 0;
			glfwGetMonitorPos(monitors[i], &xPos, &yPos);
			const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);

			if(windowPos.x > xPos && windowPos.x < xPos + mode->width &&
			   windowPos.y > yPos && windowPos.y < yPos + mode->height)
			{
				return i;
			}
		}
		return 0;
	}



	int Window::MonitorCount()
	{
		int count = 0;
		glfwGetMonitors(&count);
		return count;
	}



	float Window::PrimaryMonitorDPI()
	{
		float xScale = 0;
		float yScale = 0;
		glfwGetMonitorContentScale(glfwGetPrimaryMonitor(), &xScale, &yScale);
		return xScale;
	}



	float Window::MonitorDPI(int monitorIndex)
	{
		int count = 0;
		GLFWmonitor** monitors = glfwGetMonitors(&count);
		assert(monitorIndex < count);

		float xScale = 0;
		float yScale = 0;
		glfwGetMonitorContentScale(monitors[monitorIndex], &xScale, &yScale);
		return xScale;
	}



	//--------------------------------------------------------------------------------------------------------------------------
	// GLFW Input callbacks
	//--------------------------------------------------------------------------------------------------------------------------
	void Window::ErrorCallback(int error, const char* description) noexcept
	{
		Logger::Error("GLFW error %i: %s", error, description);
	}



	void Window::KeyCallback(GLFWwindow* handle, int key, int scancode, int action, int mods) noexcept
	{
		if(ImGui::GetIO().WantCaptureKeyboard)
			return;

		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		if(key < static_cast<int>(Input::KeyData::KeyboardCount))
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
		if(ImGui::GetIO().WantCaptureMouse)
			return;

		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		if(button < static_cast<int>(Input::KeyData::MouseCount))
			window->mNextInputState.Mouse[static_cast<size_t>(button)] = (action == GLFW_PRESS);
	}



	void Window::CursorPosCallback(GLFWwindow* handle, double xPos, double yPos) noexcept
	{
		if(ImGui::GetIO().WantCaptureMouse)
			return;

		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		window->mNextInputState.MousePos = make_float2(static_cast<float>(xPos), static_cast<float>(yPos));
	}



	void Window::CursorEnterCallback(GLFWwindow* handle, int entered) noexcept
	{
		//if(ImGui::GetIO().WantCaptureMouse)
		//	return;

		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		window->mNextInputState.MouseIsWithinWindow = !entered;
	}



	void Window::ScrollCallback(GLFWwindow* handle, double xOffset, double yOffset) noexcept
	{
		if(ImGui::GetIO().WantCaptureMouse)
			return;

		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		window->mNextInputState.MouseScroll += make_float2(static_cast<float>(xOffset), static_cast<float>(yOffset));
	}



	void Window::DropCallback(GLFWwindow* handle, int count, const char** paths) noexcept
	{
		//Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
	}
}
