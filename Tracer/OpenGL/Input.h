#pragma once

// Project
#include "Utility/Enum.h"
#include "Utility/LinearMath.h"

// C++
#include <array>
#include <cstdint>

namespace Tracer
{
	namespace Input
	{
		// key codes (GLFW_KEY_<...>)
		enum class Keys : uint32_t
		{
			Unknown         = 0,

			// Keyboard
			Space           = 32,
			Apostrophe      = 39,
			Comma           = 44,
			Minus           = 45,
			Period          = 46,
			Slash           = 47,
			Num0            = 48,
			Num1            = 49,
			Num2            = 50,
			Num3            = 51,
			Num4            = 52,
			Num5            = 53,
			Num6            = 54,
			Num7            = 55,
			Num8            = 56,
			Num9            = 57,
			Semicolon       = 59,
			Equal           = 61,
			A               = 65,
			B               = 66,
			C               = 67,
			D               = 68,
			E               = 69,
			F               = 70,
			G               = 71,
			H               = 72,
			I               = 73,
			J               = 74,
			K               = 75,
			L               = 76,
			M               = 77,
			N               = 78,
			O               = 79,
			P               = 80,
			Q               = 81,
			R               = 82,
			S               = 83,
			T               = 84,
			U               = 85,
			V               = 86,
			W               = 87,
			X               = 88,
			Y               = 89,
			Z               = 90,
			LeftBracket     = 91,
			Backslash       = 92,
			RightBracket    = 93,
			GraveAccent     = 96,
			World1          = 161,
			World2          = 162,
			Escape          = 256,
			Enter           = 257,
			Tab             = 258,
			Backspace       = 259,
			Insert          = 260,
			Delete          = 261,
			Right           = 262,
			Left            = 263,
			Down            = 264,
			Up              = 265,
			PageUp          = 266,
			PageDown        = 267,
			Home            = 268,
			End             = 269,
			CapsLock        = 280,
			ScrollLock      = 281,
			NumLock         = 282,
			PrintScreen     = 283,
			Pause           = 284,
			F1              = 290,
			F2              = 291,
			F3              = 292,
			F4              = 293,
			F5              = 294,
			F6              = 295,
			F7              = 296,
			F8              = 297,
			F9              = 298,
			F10             = 299,
			F11             = 300,
			F12             = 301,
			F13             = 302,
			F14             = 303,
			F15             = 304,
			F16             = 305,
			F17             = 306,
			F18             = 307,
			F19             = 308,
			F20             = 309,
			F21             = 310,
			F22             = 311,
			F23             = 312,
			F24             = 313,
			F25             = 314,
			KeyPad_Num0     = 320,
			KeyPad_Num1     = 321,
			KeyPad_Num2     = 322,
			KeyPad_Num3     = 323,
			KeyPad_Num4     = 324,
			KeyPad_Num5     = 325,
			KeyPad_Num6     = 326,
			KeyPad_Num7     = 327,
			KeyPad_Num8     = 328,
			KeyPad_Num9     = 329,
			KeyPad_Decimal  = 330,
			KeyPad_Divide   = 331,
			KeyPad_Multiply = 332,
			KeyPad_Subtract = 333,
			KeyPad_Add      = 334,
			KeyPad_Enter    = 335,
			KeyPad_Equal    = 336,
			LeftShift       = 340,
			LeftControl     = 341,
			LeftAlt         = 342,
			LeftSuper       = 343,
			RightShift      = 344,
			RightControl    = 345,
			RightAlt        = 346,
			RightSuper      = 347,
			Menu            = 348,

			// Mouse
			Mouse_Left,
			Mouse_Right,
			Mouse_Middle,

			// cannot be queried
			Mouse_Scroll
		};



		// modifiers
		enum class ModifierKeys : uint32_t
		{
			None  = 0x0,
			Alt   = 0x1,
			Ctrl  = 0x2,
			Shift = 0x4
		};
		ENUM_BITWISE_OPERATORS(ModifierKeys);



		// keys helpers
		enum class KeyData : std::underlying_type_t<Keys>
		{
			// keyboard
			FirstKeyboard = 0,
			LastKeyboard  = Keys::Menu,
			KeyboardCount = LastKeyboard - FirstKeyboard + 1,

			// mouse
			FirstMouse    = Keys::Mouse_Left,
			LastMouse     = Keys::Mouse_Middle,
			MouseCount    = LastMouse - FirstMouse + 1,

			// special
			FirstSpecial  = Keys::Mouse_Scroll,
			LastSpecial   = Keys::Mouse_Scroll,
			SpecialCount  = LastSpecial - FirstSpecial + 1,

			// Total
			Count         = LastMouse,
		};



		// compare Key with KeyData
		inline constexpr bool operator < (Keys key, KeyData data)
		{
			return std::underlying_type_t<Keys>(key) < std::underlying_type_t<KeyData>(data);
		}

		inline constexpr bool operator > (Keys key, KeyData data)
		{
			return std::underlying_type_t<Keys>(key) > std::underlying_type_t<KeyData>(data);
		}

		inline constexpr bool operator <= (Keys key, KeyData data)
		{
			return std::underlying_type_t<Keys>(key) <= std::underlying_type_t<KeyData>(data);
		}

		inline constexpr bool operator >= (Keys key, KeyData data)
		{
			return std::underlying_type_t<Keys>(key) >= std::underlying_type_t<KeyData>(data);
		}



		// Key type
		inline constexpr bool IsKeyboardKey(Keys key)
		{
			return key >= KeyData::FirstKeyboard && key <= KeyData::LastKeyboard;
		}

		inline constexpr bool IsMouseKey(Keys key)
		{
			return key >= KeyData::FirstMouse && key <= KeyData::LastMouse;
		}

		inline constexpr bool IsSpecialKey(Keys key)
		{
			return key >= KeyData::FirstSpecial && key <= KeyData::LastSpecial;
		}



		// Key index remapping
		inline constexpr std::underlying_type_t<Keys> MouseKeyIndex(Keys key)
		{
			assert(IsMouseKey(key));
			return std::underlying_type_t<Keys>(key) - std::underlying_type_t<KeyData>(KeyData::FirstMouse);
		}



		// Input state
		class State
		{
		public:
			std::array<bool, static_cast<size_t>(KeyData::KeyboardCount)> Keyboard = {};
			std::array<bool, static_cast<size_t>(KeyData::MouseCount)> Mouse = {};
			float2 MousePos = make_float2(0, 0);
			float2 MouseScroll = make_float2(0, 0);
			bool MouseIsWithinWindow = true;
		};
	};
}
