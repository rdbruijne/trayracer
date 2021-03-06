#pragma once

// Project
#include "OpenGL/Input.h"

// C++
#include <memory>
#include <string>
#include <vector>

// GLFW
struct GLFWwindow;

namespace Tracer
{
	class GLFramebuffer;
	class GLTexture;
	class Shader;
	class Texture;
	class Window
	{
	public:
		~Window() { Destroy(); }

		// Open/Close
		bool Open(const std::string& title, int2 resolution, bool fullscreen = false);
		void Close();
		bool IsClosed() const;
		void Destroy();

		// Title
		void SetTitle(const std::string& title);

		// Position
		int2 Position() const;
		void SetPosition(const int2& position);

		// Resolution
		int2 Resolution() const;
		void SetResolution(const int2& resolution);

		// fullscreen
		bool IsFullscreen() const;
		void SetFullscreen(bool fullscreen);

		// Render texture
		GLTexture* RenderTexture() { return mRenderTexture; }
		const GLTexture* RenderTexture() const { return mRenderTexture; }

		// Framebuffer
		GLFramebuffer* Framebuffer() { return mFramebuffer; }
		const GLFramebuffer* Framebuffer() const { return mFramebuffer; }

		std::shared_ptr<Texture> DownloadFramebuffer() const;

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
			enum class TonemapMethod
			{
				Aces,
				Filmic,
				Lottes,
				Reinhard,
				Reinhard2,
				Uchimura,
				Uncharted2
			};

			float exposure = 1.f;
			float gamma = 2.2f;
			TonemapMethod tonemap = TonemapMethod::Aces;
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

		// drag/drop
		bool HasDrops() const { return mDrops.size() > 0; }
		void ClearDrops() { mDrops.clear(); }
		std::vector<std::string> Drops() const { return mDrops; }

	private:
		// GLFW Input callbacks
		static void KeyCallback(GLFWwindow* handle, int key, int scancode, int action, int mods) noexcept;
		static void CharCallback(GLFWwindow* handle, unsigned int codepoint) noexcept;
		static void CharModsCallback(GLFWwindow* handle, unsigned int codepoint, int mods) noexcept;
		static void MouseButtonCallback(GLFWwindow* handle, int button, int action, int mods) noexcept;
		static void CursorPosCallback(GLFWwindow* handle, double xPos, double yPos) noexcept;
		static void CursorEnterCallback(GLFWwindow* handle, int entered) noexcept;
		static void ScrollCallback(GLFWwindow* handle, double xOffset, double yOffset) noexcept;
		static void DropCallback(GLFWwindow* handle, int count, const char** paths) noexcept;

		// Members
		GLFWwindow* mHandle = nullptr;
		GLTexture* mRenderTexture = nullptr;
		GLFramebuffer* mFramebuffer = nullptr;

		// post shader
		Shader* mQuadShader = nullptr;
		Shader* mTonemapShader = nullptr;
		ShaderProperties mShaderProperties;

		// Input
		Input::State mPrevInputState = Input::State();
		Input::State mCurInputState = Input::State();
		Input::State mNextInputState = Input::State();

		// backup data
		int2 mWindowPosBackup;
		int2 mWindowResBackup;

		// drag/drop
		std::vector<std::string> mDrops;
	};
}
