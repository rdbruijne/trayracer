#pragma once

// Project
#include "OpenGL/Input.h"

// C++
#include <string>

// GLFW
struct GLFWwindow;

namespace Tracer
{
	class GLTexture;
	class Shader;
	class Window
	{
	public:
		explicit Window(const std::string& title, const int2& resolution, bool fullscreen = false);
		~Window();

		// Info
		bool IsClosed() const;

		// Title
		void SetTitle(const std::string& title);

		// Position
		int2 Position() const;
		void SetPosition(const int2& position);

		// Resolution
		int2 Resolution() const;
		void SetResolution(const int2& resolution);

		// Render texture
		GLTexture* RenderTexture();
		const GLTexture* RenderTexture() const;

		// Display
		void Display();
		void SwapBuffers();

		// Input
		void UpdateInput();
		bool IsKeyDown(Input::Keys key) const;
		bool WasKeyPressed(Input::Keys key) const;
		bool IsCursorWithinWindow() const;
		float2 CursorPos() const;
		float2 Scroll() const;
		float2 CursorDelta() const;
		float2 ScrollDelta() const;

		// post shader
		struct ShaderProperties
		{
			float exposure = 1.f;
			float gamma = 2.2f;
		};
		const ShaderProperties& PostShaderProperties() const { return mShaderProperties; }
		void SetPostShaderProperties(const ShaderProperties& properties) { mShaderProperties = properties; }

		// GL
		GLFWwindow* GlfwWindow() { return mHandle; }
		const GLFWwindow* GlfwWindow() const { return mHandle; }

		// monitor info
		int CurrentMonitor() const;
		static int MonitorCount();
		static float PrimaryMonitorDPI();
		static float MonitorDPI(int monitorIndex);

	private:
		// GLFW Input callbacks
		static void ErrorCallback(int error, const char* description) noexcept;
		static void KeyCallback(GLFWwindow* handle, int key, int scancode, int action, int mods) noexcept;
		static void CharCallback(GLFWwindow* handle, unsigned int codepoint) noexcept;
		static void CharModsCallback(GLFWwindow* handle, unsigned int codepoint, int mods) noexcept;
		static void MouseButtonCallback(GLFWwindow* handle, int button, int action, int mods) noexcept;
		static void CursorPosCallback(GLFWwindow* handle, double xPos, double yPos) noexcept;
		static void CursorEnterCallback(GLFWwindow* handle, int entered) noexcept;
		static void ScrollCallback(GLFWwindow* handle, double xOffset, double yOffset) noexcept;
		static void DropCallback(GLFWwindow* handle, int count, const char** paths) noexcept;

		// Members
		int2 mResolution = make_int2(0, 0);
		GLFWwindow* mHandle = nullptr;
		GLTexture* mRenderTexture = nullptr;

		// post shader
		Shader* mShader = nullptr;
		ShaderProperties mShaderProperties;

		// Input
		Input::State mPrevInputState = Input::State();
		Input::State mCurInputState = Input::State();
		Input::State mNextInputState = Input::State();
	};
}
