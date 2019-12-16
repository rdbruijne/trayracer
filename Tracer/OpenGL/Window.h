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

	/*!
	 * @brief glfwWindow wrapper.
	 *
	 * Wrapper class for creating & managing a GLFW window.
	 */
	class Window
	{
	public:
		/*!
		 * @brief Construct a window.
		 * @param[in] title The window title.
		 * @param[in] resolution The window resolution.
		 * @param[in] fullscreen Enable a fullscreen window.
		 */
		explicit Window(const std::string& title, const int2& resolution, bool fullscreen = false);

		/*!
		 * @brief Deconstruct a window.
		 */
		~Window();

		/*!
		 * @brief Check if the window is closed
		 * @return True if the window has been closed, false otherwise.
		 */
		bool IsClosed() const;

		/*!
		 * @brief Set the title.
		 * @param[in] title The text to display in the title bar.
		 */
		void SetTitle(const std::string& title);

		/*!
		 * @brief Get the window's position.
		 * @return The window position.
		 */
		int2 GetPosition() const;

		/*!
		 * @brief Set the window position.
		 * @param[in] position The position to move the window to.
		 */
		void SetPosition(const int2& position);

		/*!
		 * @brief Get the window's resolution.
		 * @return The window resolution.
		 */
		int2 GetResolution() const;

		/*!
		 * @brief Set the window's resolution.
		 * @param[in] resolution The window resolution.
		 */
		void SetResolution(const int2& resolution);

		/*!
		 * @brief Display the window.
		 * @param[in] pixels Color buffer to display.
		 */
		void Display(const std::vector<uint32_t>& pixels);

		/*!
		 * @brief Update user input.
		 */
		void UpdateInput();


	private:
		//----------------------------------------------------------------------------------------------------------------------
		// GLFW Input callbacks
		//----------------------------------------------------------------------------------------------------------------------

		/*!
		 * @brief Handling of GLFW error messages.
		 * @param[in] error An error code.
		 * @param[in] description A UTF-8 encoded string describing the error.
		 */
		static void ErrorCallback(int error, const char* description) noexcept;

		/*!
		 * @brief Handling of key set callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] key The keyboard key that was pressed or released.
		 * @param[in] scancode The system-specific scancode of the key.
		 * @param[in] action `GLFW_PRESS`, `GLFW_RELEASE` or `GLFW_REPEAT`.
		 * @param[in] mods Bit field describing which modifier keys were held down.
		 */
		static void KeyCallback(GLFWwindow* handle, int key, int scancode, int action, int mods) noexcept;

		/*!
		 * @brief Handling of Unicode character callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] codepoint The Unicode code point of the character.
		 */
		static void CharCallback(GLFWwindow* handle, unsigned int codepoint) noexcept;

		/*!
		 * @brief Handling of Unicode character with modifiers callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] codepoint The Unicode code point of the character.
		 * @param[in] mods Bit field describing which modifier keys were held down.
		 */
		static void CharModsCallback(GLFWwindow* handle, unsigned int codepoint, int mods) noexcept;

		/*!
		 * @brief Handling of mouse button callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] button The mouse button that was pressed or released.
		 * @param[in] action One of `GLFW_PRESS` or `GLFW_RELEASE`.
		 * @param[in] mods Bit field describing which modifier keys were held down.
		 */
		static void MouseButtonCallback(GLFWwindow* handle, int button, int action, int mods) noexcept;

		/*!
		 * @brief Handling of cursor position callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] xPos The new cursor x-coordinate, relative to the left edge of the client area.
		 * @param[in] yPos The new cursor y-coordinate, relative to the top edge of the client area.
		 */
		static void CursorPosCallback(GLFWwindow* handle, double xPos, double yPos) noexcept;

		/*!
		 * @brief Handling of cursor enter/exit callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] entered `GLFW_TRUE` if the cursor entered the window's client area, or `GLFW_FALSE` if it left it.
		 */
		static void CursorEnterCallback(GLFWwindow* handle, int entered) noexcept;

		/*!
		 * @brief Handling of scroll callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] xOffset The scroll offset along the x-axis.
		 * @param[in] yOffset The scroll offset along the y-axis.
		 */
		static void ScrollCallback(GLFWwindow* handle, double xOffset, double yOffset) noexcept;

		/*!
		 * @brief Handling of drop callbacks.
		 * @param[in] handle The window that received the event.
		 * @param[in] count The number of dropped files.
		 * @param[in] paths The UTF-8 encoded file and/or directory path names.
		 */
		static void DropCallback(GLFWwindow* handle, int count, const char** paths) noexcept;



		//----------------------------------------------------------------------------------------------------------------------
		// Members
		//----------------------------------------------------------------------------------------------------------------------
		/*! Render resolution. */
		int2 mResolution = make_int2(0, 0);

		/*! Handle to the OpenGL window. */
		GLFWwindow* mHandle = nullptr;

		/*! OpenGL texture for the fullscreen quad. */
		uint32_t mGLTexture = 0;
	};
}
