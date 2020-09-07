#pragma once

#include <type_traits>

// declare trait type so that we can instantiate operators only for desired enums
template<typename enum_type>
struct enum_traits
{};

// use this to enable bitwise operators on enum classes
#define ENUM_BITWISE_OPERATORS(E)               \
    template<>                                  \
    struct enum_traits<E>                       \
    {                                           \
        static constexpr int is_bitfield = 1;   \
    };

// define template declaration to reduce code duplication
#define _BIT_ENUM_T                                                         \
    template<                                                               \
        typename E, std::enable_if_t<enum_traits<E>::is_bitfield, int> = 0, \
        typename T = std::underlying_type_t<E>>

// bitwise OR
_BIT_ENUM_T inline E operator | (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) | static_cast<T>(b)); }
_BIT_ENUM_T inline E operator | (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) | b); }
_BIT_ENUM_T inline E operator |= (E& a, E b) noexcept { return a = a | b; }
_BIT_ENUM_T inline E operator |= (E& a, T b) noexcept { return a = a | b; }

// bitwise AND
_BIT_ENUM_T inline E operator & (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) & static_cast<T>(b)); }
_BIT_ENUM_T inline E operator & (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) & b); }
_BIT_ENUM_T inline E operator &= (E& a, E b) noexcept { return a = a & b; }
_BIT_ENUM_T inline E operator &= (E& a, T b) noexcept { return a = a & b; }

// bitwise XOR
_BIT_ENUM_T inline E operator ^ (E a, E b) noexcept { return static_cast<E>(static_cast<T>(a) ^ static_cast<T>(b)); }
_BIT_ENUM_T inline E operator ^ (E a, T b) noexcept { return static_cast<E>(static_cast<T>(a) ^ b); }
_BIT_ENUM_T inline E operator ^= (E& a, E b) noexcept { return a = a ^ b; }
_BIT_ENUM_T inline E operator ^= (E& a, T b) noexcept { return a = a ^ b; }

// bitwise NOT
_BIT_ENUM_T inline E operator ~ (E a) noexcept { return static_cast<E>(~static_cast<T>(a)); }

// logical NOT
_BIT_ENUM_T inline bool operator ! (E a) noexcept { return !static_cast<T>(a); }

// logical EQ
_BIT_ENUM_T inline bool operator == (E a, T b) noexcept { return static_cast<T>(a) == b; }

// logical NEQ
_BIT_ENUM_T inline bool operator != (E a, T b) noexcept { return static_cast<T>(a) != b; }
