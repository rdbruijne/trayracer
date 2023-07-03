#pragma once

#include <type_traits>

// Enum bitwise operators
template<typename enum_type>
struct enum_bit_traits {};

#define ENUM_BITWISE_OPERATORS(E)					\
	template<>										\
	struct enum_bit_traits<E>						\
	{												\
		static constexpr bool is_bitfield = true;	\
	};



// Enum comparison operators
template<typename enum_type>
struct enum_compare_traits {};

#define ENUM_COMPARE_OPERATORS(E)					\
	template<>										\
	struct enum_compare_traits<E>					\
	{												\
		static constexpr bool has_compare = true;	\
	};



// Enum arithmetic operators
template<typename enum_type>
struct enum_arithmetic_traits {};

#define ENUM_ARITHMETIC_OPERATORS(E)				\
	template<>										\
	struct enum_arithmetic_traits<E>				\
	{												\
		static constexpr bool is_arithmetic = true;	\
	};



//------------------------------------------------------------------------------------------------------------------------------
// template declarations
//------------------------------------------------------------------------------------------------------------------------------
#define _ENUM_T(constraint)	template<typename E, typename T = std::underlying_type_t<E>, std::enable_if_t<constraint, bool> = false>
#define _BIT_ENUM_T			_ENUM_T(enum_bit_traits<E>::is_bitfield)
#define _CMP_ENUM_T			_ENUM_T(enum_compare_traits<E>::has_compare)
#define _ARIT_ENUM_T		_ENUM_T(enum_arithmetic_traits<E>::is_arithmetic)



//------------------------------------------------------------------------------------------------------------------------------
// Bitwise operators
//------------------------------------------------------------------------------------------------------------------------------
// bitwise OR
_BIT_ENUM_T constexpr E operator | (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) | static_cast<T>(b)); }
_BIT_ENUM_T constexpr E operator | (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) | b); }
_BIT_ENUM_T constexpr T operator | (T a, E b) noexcept { return a | static_cast<T>(b); }

_BIT_ENUM_T constexpr E operator |= (E& a, E b) noexcept { return a = a | b; }
_BIT_ENUM_T constexpr E operator |= (E& a, T b) noexcept { return a = a | b; }
_BIT_ENUM_T constexpr T operator |= (T& a, E b) noexcept { return a = a | b; }

// bitwise AND
_BIT_ENUM_T constexpr E operator & (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) & static_cast<T>(b)); }
_BIT_ENUM_T constexpr E operator & (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) & b); }
_BIT_ENUM_T constexpr T operator & (T a, E b) noexcept { return a & static_cast<T>(b); }

_BIT_ENUM_T constexpr E operator &= (E& a, E b) noexcept { return a = a & b; }
_BIT_ENUM_T constexpr E operator &= (E& a, T b) noexcept { return a = a & b; }
_BIT_ENUM_T constexpr E operator &= (T& a, E b) noexcept { return a = a & b; }

// bitwise XOR
_BIT_ENUM_T constexpr E operator ^ (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) ^ static_cast<T>(b)); }
_BIT_ENUM_T constexpr E operator ^ (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) ^ b); }
_BIT_ENUM_T constexpr T operator ^ (T a, E b) noexcept { return a ^ static_cast<T>(b); }

_BIT_ENUM_T constexpr E operator ^= (E& a, E b) noexcept { return a = a ^ b; }
_BIT_ENUM_T constexpr E operator ^= (E& a, T b) noexcept { return a = a ^ b; }
_BIT_ENUM_T constexpr E operator ^= (T& a, E b) noexcept { return a = a ^ b; }

// bitwise NOT
_BIT_ENUM_T constexpr E operator ~ (E a) noexcept { return static_cast<E>(~static_cast<T>(a)); }

// shift left
_ARIT_ENUM_T constexpr E operator << (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) << static_cast<T>(b)); }
_ARIT_ENUM_T constexpr E operator << (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) << b); }
_ARIT_ENUM_T constexpr T operator << (T a, E b) noexcept { return a << static_cast<T>(b); }

_ARIT_ENUM_T constexpr E operator <<= (E& a, E b) noexcept { return a = a << b; }
_ARIT_ENUM_T constexpr E operator <<= (E& a, T b) noexcept { return a = a << b; }
_ARIT_ENUM_T constexpr T operator <<= (T& a, E b) noexcept { return a = a << b; }

// shift right
_ARIT_ENUM_T constexpr E operator >> (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) >> static_cast<T>(b)); }
_ARIT_ENUM_T constexpr E operator >> (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) >> b); }
_ARIT_ENUM_T constexpr T operator >> (T a, E b) noexcept { return a >> static_cast<T>(b); }

_ARIT_ENUM_T constexpr E operator >>= (E& a, E b) noexcept { return a = a >> b; }
_ARIT_ENUM_T constexpr E operator >>= (E& a, T b) noexcept { return a = a >> b; }
_ARIT_ENUM_T constexpr T operator >>= (T& a, E b) noexcept { return a = a >> b; }



//------------------------------------------------------------------------------------------------------------------------------
// Compare operators
//------------------------------------------------------------------------------------------------------------------------------
// logical NOT
_CMP_ENUM_T constexpr bool operator ! (E a) noexcept { return !static_cast<T>(a); }

// logical EQ
_CMP_ENUM_T constexpr bool operator == (E a, T b) noexcept { return static_cast<T>(a) == b; }
_CMP_ENUM_T constexpr bool operator == (T a, E b) noexcept { return a == static_cast<T>(b); }

// logical NEQ
_CMP_ENUM_T constexpr bool operator != (E a, T b) noexcept { return !(a == b); }
_CMP_ENUM_T constexpr bool operator != (T a, E b) noexcept { return !(a == b); }

// logical GT
_CMP_ENUM_T constexpr bool operator > (E a, T b) noexcept { return static_cast<T>(a) > b; }
_CMP_ENUM_T constexpr bool operator > (T a, E b) noexcept { return a > static_cast<T>(b); }

// logical LT
_CMP_ENUM_T constexpr bool operator < (E a, T b) noexcept { return static_cast<T>(a) < b; }
_CMP_ENUM_T constexpr bool operator < (T a, E b) noexcept { return a < static_cast<T>(b); }

// logical GTE
_CMP_ENUM_T constexpr bool operator >= (E a, T b) noexcept { return !(a < b); }
_CMP_ENUM_T constexpr bool operator >= (T a, E b) noexcept { return !(a < b); }

// logical LTE
_CMP_ENUM_T constexpr bool operator <= (E a, T b) noexcept { return !(a > b); }
_CMP_ENUM_T constexpr bool operator <= (T a, E b) noexcept { return !(a > b); }



//------------------------------------------------------------------------------------------------------------------------------
// Arithmetic operators
//------------------------------------------------------------------------------------------------------------------------------
// negate
_ARIT_ENUM_T constexpr E operator + (E a) noexcept { return static_cast<E>(+static_cast<T>(a)); }
_ARIT_ENUM_T constexpr E operator - (E a) noexcept { return static_cast<E>(-static_cast<T>(a)); }

// increment
_ARIT_ENUM_T constexpr E operator ++ (E& a) noexcept { a = static_cast<E>(static_cast<T>(a) + 1); return a; }
_ARIT_ENUM_T constexpr E operator ++ (E& a, int) noexcept { E tmp = a; a = static_cast<E>(static_cast<T>(a) + 1); return tmp; }

// decrement
_ARIT_ENUM_T constexpr E operator -- (E& a) noexcept { a = static_cast<E>(static_cast<T>(a) - 1); return a; }
_ARIT_ENUM_T constexpr E operator -- (E& a, int) noexcept { E tmp = a; a = static_cast<E>(static_cast<T>(a) - 1); return tmp; }

// add
_ARIT_ENUM_T constexpr E operator + (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) + static_cast<T>(b)); }
_ARIT_ENUM_T constexpr E operator + (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) + b); }
_ARIT_ENUM_T constexpr T operator + (T a, E b) noexcept { return a + static_cast<T>(b); }

_ARIT_ENUM_T constexpr E operator += (E& a, E b) noexcept { return a = a + b; }
_ARIT_ENUM_T constexpr E operator += (E& a, T b) noexcept { return a = a + b; }
_ARIT_ENUM_T constexpr T operator += (T& a, E b) noexcept { return a = a + b; }

// subtract
_ARIT_ENUM_T constexpr E operator - (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) - static_cast<T>(b)); }
_ARIT_ENUM_T constexpr E operator - (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) - b); }
_ARIT_ENUM_T constexpr T operator - (T a, E b) noexcept { return a - static_cast<T>(b); }

_ARIT_ENUM_T constexpr E operator -= (E& a, E b) noexcept { return a = a - b; }
_ARIT_ENUM_T constexpr E operator -= (E& a, T b) noexcept { return a = a - b; }
_ARIT_ENUM_T constexpr T operator -= (T& a, E b) noexcept { return a = a - b; }

// multiply
_ARIT_ENUM_T constexpr E operator * (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) * static_cast<T>(b)); }
_ARIT_ENUM_T constexpr E operator * (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) * b); }
_ARIT_ENUM_T constexpr T operator * (T a, E b) noexcept { return a * static_cast<T>(b); }

_ARIT_ENUM_T constexpr E operator *= (E& a, E b) noexcept { return a = a * b; }
_ARIT_ENUM_T constexpr E operator *= (E& a, T b) noexcept { return a = a * b; }
_ARIT_ENUM_T constexpr T operator *= (T& a, E b) noexcept { return a = a * b; }

// divide
_ARIT_ENUM_T constexpr E operator / (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) / static_cast<T>(b)); }
_ARIT_ENUM_T constexpr E operator / (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) / b); }
_ARIT_ENUM_T constexpr T operator / (T a, E b) noexcept { return a / static_cast<T>(b); }

_ARIT_ENUM_T constexpr E operator /= (E& a, E b) noexcept { return a = a / b; }
_ARIT_ENUM_T constexpr E operator /= (E& a, T b) noexcept { return a = a / b; }
_ARIT_ENUM_T constexpr T operator /= (T& a, E b) noexcept { return a = a / b; }

// modulo
_ARIT_ENUM_T constexpr E operator % (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) % static_cast<T>(b)); }
_ARIT_ENUM_T constexpr E operator % (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) % b); }
_ARIT_ENUM_T constexpr T operator % (T a, E b) noexcept { return a % static_cast<T>(b); }

_ARIT_ENUM_T constexpr E operator %= (E& a, E b) noexcept { return a = a % b; }
_ARIT_ENUM_T constexpr E operator %= (E& a, T b) noexcept { return a = a % b; }
_ARIT_ENUM_T constexpr T operator %= (T& a, E b) noexcept { return a = a % b; }

