namespace Tracer
{
	template<typename iter>
	std::string Join(iter first, iter last, const std::string& separator)
	{
		if(first == last)
			return {};

		using T = decltype(*first);

		if constexpr (std::is_convertible_v<T, std::string>)
		{
			std::string result = *first;
			while(++first != last)
			{
				result.append(separator);
				result.append(*first);
			}
			return result;
		}
		else
		{
			std::ostringstream os;
			os << *first;
			while(++first != last)
			{
				os << separator.c_str() << *first;
			}
			return os.str();
		}
	}
}
