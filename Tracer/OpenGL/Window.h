#pragma once

// OptiX
#include "Optix7.h"

// C++
#include <string>
#include <vector>

// GLFW
struct GLFWwindow;

namespace Tracer
{
	class PostShader;
	class Window
	{
	public:
		explicit Window(const std::string& title, const int2& resolution, bool fullscreen = false);
		~Window();

		// Info
		bool IsClosed() const;

		// Title
		void SetTitle(const std::string& title);

		// Resolution
		int2 GetPosition() const;
		void SetPosition(const int2& position);

		// Position
		int2 GetResolution() const;
		void SetResolution(const int2& resolution);

		// Display
		void Display(const std::vector<uint32_t>& pixels);

		// Input
		void UpdateInput();

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
		uint32_t mGLTexture = 0;
	};
}
