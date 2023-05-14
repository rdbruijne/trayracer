#pragma once

#include <string>
#include <vector>

namespace Tracer
{
	// string manipulation
	std::string format(const char* fmt, ...);

	// capitalization
	std::string ToLower(const std::string& str);
	std::string ToUpper(const std::string& str);

	// string splitting
	std::vector<std::string> Split(const std::string& str, char delimiter, bool skipEmpty = false);
	std::vector<std::string> Split(const std::string& str, const std::string& delimiters, bool skipEmpty = false);

	// join strings
	template<typename iter>
	std::string Join(iter first, iter last, const std::string& separator);

	// trim
	void TrimFront(std::string& str);
	void TrimBack(std::string& str);
	inline void Trim(std::string& str)
	{
		TrimFront(str);
		TrimBack(str);
	}

	inline std::string TrimmedFront(std::string& str) { std::string s = str; TrimFront(s); return s; }
	inline std::string TrimmedBack(std::string& str) { std::string s = str; TrimBack(s); return s; }
	inline std::string Trimmed(std::string& str) { std::string s = str; Trim(s); return s; }

	// replacement
	std::string Remove(const std::string& str, char toRemove);
	std::string Remove(const std::string& str, const std::string& toRemove);
	std::string Remove(const std::string& str, const std::vector<char>& toRemove);
	std::string Remove(const std::string& str, const std::vector<std::string>& toRemove);

	std::string Replace(const std::string& str, char from, char to);
	std::string Replace(const std::string& str, const std::string& from, const std::string& to);

	// stringify
	std::string TimeString(int64_t elapsedNs);
	std::string ThousandSeparators(uint64_t val, const std::string& separator = ",");

	////////////////////////////////////////////////////////////////
	// implementations
	////////////////////////////////////////////////////////////////
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
