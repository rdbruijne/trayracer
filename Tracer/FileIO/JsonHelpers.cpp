#include "JsonHelpers.h"

// Project
#include "Utility/LinearMath.h"

using namespace rapidjson;

namespace Tracer
{
	//----------------------------------------------------------------------------------------------------------------------
	// Read helpers
	//----------------------------------------------------------------------------------------------------------------------
	bool Read(const Value& jsonValue, const std::string_view& memberName, bool& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsBool())
			return false;

		result = val.GetBool();
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, float& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsNumber())
			return false;

		result = val.GetFloat();
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, float2& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsArray() || val.Size() != 2 || !val[0].IsNumber() || !val[1].IsNumber())
			return false;

		result = make_float2(val[0].GetFloat(), val[1].GetFloat());
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, float3& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsArray() || val.Size() != 3 || !val[0].IsNumber() || !val[1].IsNumber() || !val[2].IsNumber())
			return false;

		result = make_float3(val[0].GetFloat(), val[1].GetFloat(), val[2].GetFloat());
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, float4& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsArray() || val.Size() != 4 || !val[0].IsNumber() || !val[1].IsNumber() || !val[2].IsNumber() || !val[3].IsNumber())
			return false;

		result = make_float4(val[0].GetFloat(), val[1].GetFloat(), val[2].GetFloat(), val[3].GetFloat());
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, float3x4& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsArray() || val.Size() != 12)
			return false;

		float m[12] = {};
		for(SizeType i = 0; i < val.Size(); i++)
		{
			if(!val[i].IsNumber())
				return false;
			m[i] = val[i].GetFloat();
		}

		result = make_float3x4(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]);
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, int& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsNumber())
			return false;

		result = val.GetInt();
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, uint32_t& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsNumber())
			return false;

		result = val.GetUint();
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, int2& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsArray() || val.Size() != 2 || !val[0].IsNumber() || !val[1].IsNumber())
			return false;

		result = make_int2(val[0].GetInt(), val[1].GetInt());
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, int3& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsArray() || val.Size() != 3 || !val[0].IsNumber() || !val[1].IsNumber() || !val[2].IsNumber())
			return false;

		result = make_int3(val[0].GetInt(), val[1].GetInt(), val[2].GetInt());
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, int4& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsArray() || val.Size() != 4 || !val[0].IsNumber() || !val[1].IsNumber() || !val[2].IsNumber() || !val[3].IsNumber())
			return false;

		result = make_int4(val[0].GetInt(), val[1].GetInt(), val[2].GetInt(), val[3].GetInt());
		return true;
	}



	bool Read(const Value& jsonValue, const std::string_view& memberName, std::string& result)
	{
		if(!jsonValue.HasMember(memberName.data()))
			return false;

		const Value& val = jsonValue[memberName.data()];
		if(!val.IsString())
			return false;

		result = val.GetString();
		return true;
	}



	//----------------------------------------------------------------------------------------------------------------------
	// Write helpers
	//----------------------------------------------------------------------------------------------------------------------
	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, bool val)
	{
		Value v = Value(kObjectType);
		v.SetBool(val);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, v, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, float val)
	{
		Value v = Value(kObjectType);
		v.SetFloat(val);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, v, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const float2& val)
	{
		Value jsonVector = Value(kArrayType);
		jsonVector.PushBack(val.x, allocator);
		jsonVector.PushBack(val.y, allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, jsonVector, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const float3& val)
	{
		Value jsonVector = Value(kArrayType);
		jsonVector.PushBack(val.x, allocator);
		jsonVector.PushBack(val.y, allocator);
		jsonVector.PushBack(val.z, allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, jsonVector, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const float4& val)
	{
		Value jsonVector = Value(kArrayType);
		jsonVector.PushBack(val.x, allocator);
		jsonVector.PushBack(val.y, allocator);
		jsonVector.PushBack(val.z, allocator);
		jsonVector.PushBack(val.w, allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, jsonVector, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const float3x4& val)
	{
		Value jsonMatrix = Value(kArrayType);
		jsonMatrix.PushBack(val.x.x, allocator);
		jsonMatrix.PushBack(val.x.y, allocator);
		jsonMatrix.PushBack(val.x.z, allocator);
		jsonMatrix.PushBack(val.y.x, allocator);
		jsonMatrix.PushBack(val.y.y, allocator);
		jsonMatrix.PushBack(val.y.z, allocator);
		jsonMatrix.PushBack(val.z.x, allocator);
		jsonMatrix.PushBack(val.z.y, allocator);
		jsonMatrix.PushBack(val.z.z, allocator);
		jsonMatrix.PushBack(val.tx, allocator);
		jsonMatrix.PushBack(val.ty, allocator);
		jsonMatrix.PushBack(val.tz, allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, jsonMatrix, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator,const std::string_view& memberName, int val)
	{
		Value v = Value(kObjectType);
		v.SetInt(val);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, v, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator,const std::string_view& memberName, uint32_t val)
	{
		Value v = Value(kObjectType);
		v.SetUint(val);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, v, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const int2& val)
	{
		Value jsonVector = Value(kArrayType);
		jsonVector.PushBack(val.x, allocator);
		jsonVector.PushBack(val.y, allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, jsonVector, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const int3& val)
	{
		Value jsonVector = Value(kArrayType);
		jsonVector.PushBack(val.x, allocator);
		jsonVector.PushBack(val.y, allocator);
		jsonVector.PushBack(val.z, allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, jsonVector, allocator);
	}



	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const int4& val)
	{
		Value jsonVector = Value(kArrayType);
		jsonVector.PushBack(val.x, allocator);
		jsonVector.PushBack(val.y, allocator);
		jsonVector.PushBack(val.z, allocator);
		jsonVector.PushBack(val.w, allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, jsonVector, allocator);
	}


	void Write(Value& jsonValue, Document::AllocatorType& allocator, const std::string_view& memberName, const std::string& val)
	{
		Value v = Value(kObjectType);
		v.SetString(val.c_str(), static_cast<SizeType>(val.length()), allocator);

		Value key = Value(memberName.data(), allocator);
		jsonValue.AddMember(key, v, allocator);
	}
}
