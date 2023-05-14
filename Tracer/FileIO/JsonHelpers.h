#pragma once

// RapidJson
#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#pragma warning(disable: 4619) // #pragma warning : there is no warning number 'number'
#pragma warning(disable: 5054) // operator 'operator-name': deprecated between enumerations of different types
#include "RapidJson/document.h"
#include "RapidJson/istreamwrapper.h"
#include "RapidJson/ostreamwrapper.h"
#include "RapidJson/prettywriter.h"
#pragma warning(pop)

// Magic Enum
#pragma warning(push)
#pragma warning(disable: 4346) // 'name' : dependent name is not a type
#pragma warning(disable: 4626) // 'derived class' : assignment operator was implicitly defined as deleted because a base class assignment operator is inaccessible or deleted
#pragma warning(disable: 5027) // 'type': move assignment operator was implicitly defined as deleted
#include "magic_enum/magic_enum.hpp"
#pragma warning(pop)

// C++
#include <string>
#include <string_view>

struct float2;
struct float3;
struct float4;
struct int2;
struct int3;
struct int4;
namespace Tracer
{
	struct float3x4;
}


namespace Tracer
{
	// Read helpers
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, bool& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, float& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, float2& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, float3& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, float4& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, float3x4& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, int& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, int2& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, int3& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, int4& result);
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, std::string& result);

	template<typename Enum, std::enable_if_t<std::is_enum_v<Enum>, bool> = true>
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, Enum& result);

	// Write helpers
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, bool val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, float val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const float2& val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const float3& val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const float4& val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const float3x4& val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator,const std::string_view& memberName, int val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const int2& val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const int3& val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const int4& val);
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, const std::string& val);

	template<typename Enum, std::enable_if_t<std::is_enum_v<Enum>, bool> = true>
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, Enum val);
}

#include "JsonHelpers.inl"
