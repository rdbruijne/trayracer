namespace Tracer
{
	template<typename Enum, typename>
	bool Read(const rapidjson::Value& jsonValue, const std::string_view& memberName, Enum& result)
	{
		std::string s;
		if(!Read(jsonValue, memberName, s))
			return false;

		std::optional<Enum> e = magic_enum::enum_cast<Enum>(s);
		if(!e.has_value())
			return false;

		result = e.value();
		return true;
	}



	template<typename Enum, typename>
	void Write(rapidjson::Value& jsonValue, rapidjson::Document::AllocatorType& allocator, const std::string_view& memberName, Enum val)
	{
		const std::string s = std::string(magic_enum::enum_name(val));
		Write(jsonValue, allocator, memberName, s);
	}
}
