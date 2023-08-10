#pragma once

// Project
#include "Utility/Enum.h"
#include "Utility/LinearMath.h"

// C++
#include <array>
#include <cstdint>
#include <variant>

// https://www.glfw.org/docs/3.3/input_guide.html

namespace Tracer
{
	namespace Input
	{
		// key codes (GLFW_KEY_<...>)
		enum class Keys : uint16_t
		{
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

			_Count          = Menu
		};



		// modifiers (GLFW_MOD_<...>)
		enum class Modifiers : uint8_t
		{
			None     = 0x0000,
			Shift    = 0x0001, // If this bit is set one or more Shift keys were held down.
			Ctrl     = 0x0002, // If this bit is set one or more Control keys were held down.
			Alt      = 0x0004, // If this bit is set one or more Alt keys were held down.
			Super    = 0x0008, // If this bit is set one or more Super keys were held down.
			CapsLock = 0x0010, // If this bit is set the Caps Lock key is enabled and the GLFW_LOCK_KEY_MODS input mode is set.
			NumLock  = 0x0020  // If this bit is set the Num Lock key is enabled and the GLFW_LOCK_KEY_MODS input mode is set.
		};
		ENUM_BITWISE_OPERATORS(Modifiers);



		// key codes (GLFW_MOUSE_BUTTON_<...>)
		enum class MouseButtons : uint8_t
		{
			Button1,
			Button2,
			Button3,
			Button4,
			Button5,
			Button6,
			Button7,
			Button8,

			Left   = Button1,
			Right  = Button2,
			Middle = Button3,

			_Count = Button8
		};



		enum class MouseScroll : uint8_t
		{
			Horizontal,
			Vertical,

			_Count = 2
		};



		enum class MouseMove : uint8_t
		{
			Horizontal,
			Vertical,

			_Count = 2
		};



#pragma warning(push)
#pragma warning(disable: 4324) // warning C4324: 'Tracer::Input::State': structure was padded due to alignment specifier
		// Input state
		class State
		{
		public:
			bool MouseIsWithinWindow = true;
			float2 MousePos = make_float2(0, 0);
			float2 MouseScroll = make_float2(0, 0);
			Modifiers Modifiers = Modifiers::None;

			std::array<bool, static_cast<size_t>(MouseButtons::_Count)> Mouse = {};
			std::array<bool, static_cast<size_t>(Keys::_Count)> Keyboard = {};
		};
#pragma warning(pop)



		// Keybinding
		class Keybind
		{
		public:
			enum class Types : uint8_t
			{
				None = 0,
				Key,
				MouseButton,
				MouseScroll,
				MouseMove
			};

			Keybind() = default;

			explicit Keybind(Keys key, Modifiers modifiers = Modifiers::None) :
				mType(Types::Key), mModifiers(modifiers), mBinding(static_cast<uint16_t>(key)) {}
			explicit Keybind(MouseButtons key, Modifiers modifiers = Modifiers::None) :
				mType(Types::MouseButton), mModifiers(modifiers), mBinding(static_cast<uint16_t>(key)) {}
			explicit Keybind(MouseScroll key, Modifiers modifiers = Modifiers::None) :
				mType(Types::MouseScroll), mModifiers(modifiers), mBinding(static_cast<uint16_t>(key)) {}
			explicit Keybind(MouseMove key, Modifiers modifiers = Modifiers::None) :
				mType(Types::MouseMove), mModifiers(modifiers), mBinding(static_cast<uint16_t>(key)) {}

			// basic data
			Types Type() const { return mType; }
			Modifiers Mods() const { return mModifiers; }
			uint16_t Binding() const { return mBinding; }

			// binding
			template<typename TYPE>
			bool Is() const noexcept
			{
				if constexpr (std::is_same<TYPE, Keys>())
					return mType == Types::Key;
				if constexpr (std::is_same<TYPE, MouseButtons>())
					return mType == Types::MouseButton;
				if constexpr (std::is_same<TYPE, MouseScroll>())
					return mType == Types::MouseScroll;
				if constexpr (std::is_same<TYPE, MouseMove>())
					return mType == Types::MouseMove;
			}

			template<typename TYPE>
			TYPE Get() const
			{
				assert(Is<TYPE>());
				return static_cast<TYPE>(mBinding);
			}

			template<typename TYPE>
			bool TryGet(TYPE& val) const
			{
				if(Is<TYPE>())
				{
					val = static_cast<TYPE>(mBinding);
					return true;
				}
				return false;
			}

			// comparison
			friend bool operator == (const Keybind& a, const Keybind& b)
			{
				return (a.mType == b.mType) && (a.mModifiers == b.mModifiers) && (a.mBinding == b.mBinding);
			}
			friend bool operator != (const Keybind& a, const Keybind& b) { return !(a == b); }

		private:
			Types mType = Types::None; // u8
			Modifiers mModifiers = Modifiers::None; // u8
			uint16_t mBinding; // u16
		};
	};
}
