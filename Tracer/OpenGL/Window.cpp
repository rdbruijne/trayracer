#include "Window.h"

// Project
#include "Logging/Logger.h"
#include "OpenGL/GLFramebuffer.h"
#include "OpenGL/GLTexture.h"
#include "OpenGL/Shader.h"
#include "Resources/Texture.h"
#include "Utility/Errors.h"
#include "Utility/LinearMath.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// ImGUI
#include "imgui/imgui.h"

// C++
#include <cassert>
#include <stdexcept>

// Windows
#ifdef APIENTRY
#undef APIENTRY
#endif

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace Tracer
{
	void Window::Open(const std::string& title, int2 resolution, bool fullscreen)
	{
		if(!IsClosed())
			return;

		// clean state
		Destroy();

		// create window
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		if(fullscreen && (resolution.x <= 0 || resolution.y <= 0))
		{
			const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			resolution = make_int2(mode->width, mode->height);
		}

		mHandle = glfwCreateWindow(resolution.x, resolution.y, title.c_str(), fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

		if(!mHandle)
			FatalError("Failed to create glfw window");

		// user pointer setup
		glfwSetWindowUserPointer(mHandle, this);
		glfwMakeContextCurrent(mHandle);

		// enable VSync
		glfwSwapInterval(0);

		// set window callbacks
		glfwSetKeyCallback(mHandle, Window::KeyCallback);
		glfwSetCharCallback(mHandle, Window::CharCallback);
		glfwSetCharModsCallback(mHandle, Window::CharModsCallback);
		glfwSetMouseButtonCallback(mHandle, Window::MouseButtonCallback);
		glfwSetCursorPosCallback(mHandle, Window::CursorPosCallback);
		glfwSetCursorEnterCallback(mHandle, Window::CursorEnterCallback);
		glfwSetScrollCallback(mHandle, Window::ScrollCallback);
		glfwSetDropCallback(mHandle, Window::DropCallback);

		// set input modes
		glfwSetInputMode(mHandle, GLFW_LOCK_KEY_MODS, GLFW_TRUE);

		// init GL
		if(glewInit() != GLEW_OK)
			FatalError("Failed to init glew");

		// size buffers
		SetResolution(resolution);

		// backup resolution & size
		mWindowPosBackup = Position();
		mWindowResBackup = Resolution();

		// settings
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		// init shader
		mQuadShader = new Shader("quad");
	}



	void Window::Close()
	{
		if(mHandle)
			glfwSetWindowShouldClose(mHandle, GLFW_TRUE);
	}



	bool Window::IsClosed() const
	{
		return (!mHandle) || (glfwWindowShouldClose(mHandle) != 0);
	}



	void Window::Destroy()
	{
		if(mQuadShader)
		{
			delete mQuadShader;
			mQuadShader = nullptr;
		}

		if(mRenderTexture)
		{
			delete mRenderTexture;
			mRenderTexture = nullptr;
		}

		if(mFramebuffers)
		{
			delete mFramebuffers[0];
			delete mFramebuffers[1];
			mFramebuffers[0] = nullptr;
			mFramebuffers[1] = nullptr;
		}

		if (mHandle)
		{
			glfwDestroyWindow(mHandle);
			mHandle = nullptr;
		}
	}



	void Window::SetTitle(const std::string& title)
	{
		if(!IsClosed())
			glfwSetWindowTitle(mHandle, title.c_str());
	}



	int2 Window::Position() const
	{
		int2 position = make_int2(0, 0);
		if(!IsClosed())
			glfwGetWindowPos(mHandle, &position.x, &position.y);
		return position;
	}



	void Window::SetPosition(const int2& position)
	{
		if(!IsClosed())
			glfwSetWindowPos(mHandle, position.x, position.y);
	}



	int2 Window::Resolution() const
	{
		int2 resolution = make_int2(0, 0);
		if(!IsClosed())
		{
			glfwGetWindowSize(mHandle, &resolution.x, &resolution.y);

			const float dpiScale = DpiScale();
			const int scaledResX = static_cast<int>(static_cast<float>(resolution.x) / dpiScale);
			const int scaledResY = static_cast<int>(static_cast<float>(resolution.y) / dpiScale);
			resolution = make_int2(scaledResX, scaledResY);
		}
		return resolution;
	}



	void Window::SetResolution(const int2& resolution)
	{
		if(IsClosed())
			return;

		if(!mRenderTexture || mRenderTexture->Resolution() != resolution)
		{
			delete mRenderTexture;
			mRenderTexture = new GLTexture(resolution, GLTexture::Types::Float4);
		}

		if(!mFramebuffers[0] || mFramebuffers[0]->Resolution() != resolution)
		{
			delete mFramebuffers[0];
			delete mFramebuffers[1];
			mFramebuffers[0] = new GLFramebuffer(resolution);
			mFramebuffers[1] = new GLFramebuffer(resolution);
		}

		const float dpiScale = DpiScale();
		const int scaledResX = static_cast<int>(static_cast<float>(resolution.x) * dpiScale);
		const int scaledResY = static_cast<int>(static_cast<float>(resolution.y) * dpiScale);
		const int2 dpiRes = make_int2(scaledResX, scaledResY);
		glfwSetWindowSize(mHandle, dpiRes.x, dpiRes.y);

		glfwMakeContextCurrent(mHandle);

		// viewport
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 1, 0, 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glViewport(0, 0, dpiRes.x, dpiRes.y);
	}



	bool Window::IsFullscreen() const
	{
		return IsClosed() ? false : glfwGetWindowMonitor(mHandle) != nullptr;
	}



	void Window::SetFullscreen(bool fullscreen)
	{
		if(IsClosed())
			return;

		if(fullscreen == IsFullscreen())
			return;

		if(fullscreen)
		{
			// backup position & size
			mWindowPosBackup = Position();
			mWindowResBackup = Resolution();

			// get monitor resolution
			const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

			// switch to full screen
			glfwSetWindowMonitor(mHandle, glfwGetPrimaryMonitor(), 0, 0, mode->width, mode->height, 0);
			SetResolution(make_int2(mode->width, mode->height));
		}
		else
		{
			// restore last size & position
			glfwSetWindowMonitor(mHandle, nullptr, std::max(100, mWindowPosBackup.x), std::max(100, mWindowPosBackup.y),
								 mWindowResBackup.x, mWindowResBackup.y, 0);
			SetResolution(mWindowResBackup);
		}
	}



	std::shared_ptr<Texture> Window::DownloadFramebuffer() const
	{
		std::vector<float4> pixels;
		Framebuffer()->Texture()->Download(pixels);
		return std::make_shared<Texture>("", Resolution(), pixels);
	}



	void Window::Display()
	{
		if(IsClosed())
			return;

		glfwMakeContextCurrent(mHandle);

		// DPI fix
		SetResolution(Resolution());

		glEnable(GL_TEXTURE_2D);
		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);

		// execute the shaders
		int bufferIx = 0;
		GLTexture* prevTex = mRenderTexture;
		for(const std::shared_ptr<Shader>& shader : mPostStack)
		{
			if(!shader->IsEnabled() || !shader->IsValid())
				continue;

			mFramebuffers[bufferIx]->Bind();
			shader->Bind();
			shader->ApplyUniforms();
			shader->Set("convergeBuffer", 0, prevTex);
			glDrawArrays(GL_TRIANGLES, 0, 3);
			shader->Unbind();
			mFramebuffers[bufferIx]->Unbind();

			prevTex = mFramebuffers[bufferIx]->Texture();
			bufferIx ^= 1;
		}

		// draw the framebuffer to the screen
		mQuadShader->Bind();
		mQuadShader->ApplyUniforms();
		mQuadShader->Set("convergeBuffer", 0, prevTex);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		mQuadShader->Unbind();

		glFinish();
	}



	void Window::SwapBuffers()
	{
		if(!IsClosed())
			glfwSwapBuffers(mHandle);
	}



	//
	// Input
	//
	void Window::UpdateInput()
	{
		if(!IsClosed())
		{
			glfwMakeContextCurrent(mHandle);
			glfwPollEvents();

			mPrevInputState = mCurInputState;
			mCurInputState  = mNextInputState;
		}
	}



	// mouse events will be based on scroll/move, keys will be 0 or 1
	float Window::CheckInput(Input::Keybind keybind) const
	{
		// keyboard key
		Input::Keys keyboardKey;
		if(keybind.TryGet(keyboardKey))
			return IsDown(keyboardKey) && IsDown(keybind.Mods()) ? 1.f : 0.f;

		// mouse key
		Input::MouseButtons MouseButtons;
		if(keybind.TryGet(MouseButtons))
			return IsDown(MouseButtons) && IsDown(keybind.Mods()) ? 1.f : 0.f;

		// mouse scroll
		Input::MouseScroll mouseScroll;
		if(keybind.TryGet(mouseScroll))
			return mouseScroll == Input::MouseScroll::Vertical ? ScrollDelta().y : ScrollDelta().x;

		// mouse move
		Input::MouseMove mouseMove;
		if(keybind.TryGet(mouseMove))
			return mouseMove == Input::MouseMove::Vertical ? CursorDelta().y : CursorDelta().x;

		Logger::Error("Invalid keybind provided.");
		assert(false);
		return 0.f;
	}



	bool Window::IsDown(Input::Keys key) const
	{
		return mCurInputState.Keyboard[static_cast<size_t>(key)];
	}



	bool Window::IsDown(Input::Modifiers keys) const
	{
		return HasFlag(mCurInputState.Modifiers, keys);
	}



	bool Window::IsDown(Input::MouseButtons button) const
	{
		return mCurInputState.Mouse[static_cast<size_t>(button)];
	}


	
	bool Window::IsDown(Input::Keybind keybind) const
	{
		// keyboard key
		Input::Keys keyboardKey;
		if(keybind.TryGet(keyboardKey))
			return IsDown(keyboardKey) && IsDown(keybind.Mods());

		// mouse key
		Input::MouseButtons MouseButtons;
		if(keybind.TryGet(MouseButtons))
			return IsDown(MouseButtons) && IsDown(keybind.Mods());

		return false;
	}



	bool Window::WasPressed(Input::Keys key) const
	{
		const size_t keyIx = static_cast<size_t>(key);
		return mCurInputState.Keyboard[keyIx] && !mPrevInputState.Keyboard[keyIx];
	}



	bool Window::WasPressed(Input::Modifiers keys) const
	{
		return HasFlag(mCurInputState.Modifiers, keys) && !HasFlag(mPrevInputState.Modifiers, keys);
	}



	bool Window::WasPressed(Input::MouseButtons button) const
	{
		const size_t buttonIx = static_cast<size_t>(button);
		return mCurInputState.Mouse[buttonIx] && !mPrevInputState.Mouse[buttonIx];
	}



	bool Window::WasPressed(Input::Keybind keybind) const
	{
		// keyboard key
		Input::Keys keyboardKey;
		if(keybind.TryGet(keyboardKey))
			return WasPressed(keyboardKey) && IsDown(keybind.Mods());

		// mouse key
		Input::MouseButtons MouseButtons;
		if(keybind.TryGet(MouseButtons))
			return WasPressed(MouseButtons) && IsDown(keybind.Mods());

		return false;
	}



	bool Window::IsCursorWithinWindow() const
	{
		return mCurInputState.MouseIsWithinWindow;
	}



	float2 Window::CursorPos() const
	{
		return mCurInputState.MousePos / DpiScale();
	}



	float2 Window::CursorDelta() const
	{
		return mCurInputState.MousePos - mPrevInputState.MousePos;
	}



	float2 Window::ScrollPos() const
	{
		return mCurInputState.MouseScroll;
	}



	float2 Window::ScrollDelta() const
	{
		return mCurInputState.MouseScroll - mPrevInputState.MouseScroll;
	}



	float Window::DpiScale() const
	{
		return MonitorDpiScale(CurrentMonitor());
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



	float Window::PrimaryMonitorDpiScale()
	{
		float xScale = 0;
		float yScale = 0;
		glfwGetMonitorContentScale(glfwGetPrimaryMonitor(), &xScale, &yScale);
		return xScale;
	}



	float Window::MonitorDpiScale(int monitorIndex)
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
	void Window::KeyCallback(GLFWwindow* handle, int key, int /*scancode*/, int action, int mods) noexcept
	{
		if(!ImGui::GetIO().WantCaptureKeyboard)
		{
			Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
			if(key < static_cast<int>(Input::Keys::_Count))
				window->mNextInputState.Keyboard[static_cast<size_t>(key)] = (action != GLFW_RELEASE);
			window->mNextInputState.Modifiers = static_cast<Input::Modifiers>(mods);
		}
	}



	// receives Unicode code points for key events that would have led to regular text input
	void Window::CharCallback(GLFWwindow* /*handle*/, unsigned int /*codepoint*/) noexcept
	{
		//Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
	}



	void Window::CharModsCallback(GLFWwindow* /*handle*/, unsigned int /*codepoint*/, int /*mods*/) noexcept
	{
		//Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
	}



	void Window::MouseButtonCallback(GLFWwindow* handle, int button, int action, int /*mods*/) noexcept
	{
		if(!ImGui::GetIO().WantCaptureMouse)
		{
			Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
			window->mNextInputState.Mouse[static_cast<size_t>(button)] = (action != GLFW_RELEASE);
		}
	}



	void Window::CursorPosCallback(GLFWwindow* handle, double xPos, double yPos) noexcept
	{
		if(!ImGui::GetIO().WantCaptureMouse)
		{
			Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
			window->mNextInputState.MousePos = make_float2(static_cast<float>(xPos), static_cast<float>(yPos));
		}
	}



	void Window::CursorEnterCallback(GLFWwindow* handle, int entered) noexcept
	{
		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		window->mNextInputState.MouseIsWithinWindow = (entered != 0);
	}



	void Window::ScrollCallback(GLFWwindow* handle, double xOffset, double yOffset) noexcept
	{
		if(!ImGui::GetIO().WantCaptureMouse)
		{
			Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
			window->mNextInputState.MouseScroll += make_float2(static_cast<float>(xOffset), static_cast<float>(yOffset));
		}
	}



	void Window::DropCallback(GLFWwindow* handle, int count, const char** paths) noexcept
	{
		Window* const window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(handle));
		for(int i = 0; i < count; i++)
			window->mDrops.push_back(paths[i]);
	}
}
