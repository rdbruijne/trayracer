#include "Strings.h"

// C++
#include <cstdarg>
#include <algorithm>
#include <sstream>



namespace Tracer
{
	std::string format(const char* fmt, ...)
	{
		size_t n = strlen(fmt);
		int final_n = -1;

		std::unique_ptr<char[]> formatted;
		va_list ap;
		do
		{
			n *= 2;
			formatted.reset(new char[n + 1]);
			va_start(ap, fmt);
			final_n = vsnprintf_s(formatted.get(), n - 1, n - 2, fmt, ap);
			va_end(ap);
		} while(final_n < 0);
		return std::string(formatted.get());
	}



	std::string ToLower(const std::string& str)
	{
		std::string s = str;
		std::transform(s.begin(), s.end(), s.begin(), [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
		return s;
	}



	std::string ToUpper(const std::string& str)
	{
		std::string s = str;
		std::transform(s.begin(), s.end(), s.begin(), [](char c) { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
		return s;
	}



	std::vector<std::string> Split(const std::string& str, char delimiter, bool skipEmpty)
	{
		std::vector<std::string> result;
		if(str.empty())
			return result;

		// up to first occurrence of delimeter
		std::string::size_type lastPos = str.find(delimiter, 0);
		if(!skipEmpty || lastPos > 0)
			result.emplace_back(str.substr(0, lastPos));

		// split the rest
		while(lastPos != std::string::npos)
		{
			lastPos++;
			const std::string::size_type pos = str.find(delimiter, lastPos);
			if(!skipEmpty || (lastPos < pos && pos != str.size()))
				result.push_back(str.substr(lastPos, pos - lastPos));
			lastPos = pos;
		}

		return result;
	}



	std::vector<std::string> Split(const std::string& str, const std::string& delimiters, bool skipEmpty)
	{
		std::vector<std::string> result;
		if(str.empty())
			return result;

		// up to first occurrence of delimeter
		std::string::size_type lastPos = str.find_first_of(delimiters, 0);
		if(!skipEmpty || lastPos > 0)
			result.emplace_back(str.substr(0, lastPos));

		// split the rest
		while(lastPos != std::string::npos)
		{
			lastPos++;
			const std::string::size_type pos = str.find_first_of(delimiters, lastPos);
			if(!skipEmpty || (lastPos < pos && pos != str.size()))
				result.push_back(str.substr(lastPos, pos - lastPos));
			lastPos = pos;
		}

		return result;
	}



	void TrimFront(std::string& str)
	{
		str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](char c) { return !std::isspace(c); }));
	}



	void TrimBack(std::string& str)
	{
		str.erase(std::find_if(str.rbegin(), str.rend(), [](char c) { return !std::isspace(c); }).base(), str.end());
	}



	std::string Remove(const std::string& str, char toRemove)
	{
		std::string result;
		result.reserve(str.size());
		for(char c : str)
		{
			if(c != toRemove)
				result.push_back(c);
		}
		return result;
	}



	std::string Remove(const std::string& str, const std::string& toRemove)
	{
		std::string result;
		result.reserve(str.size());

		// up to first occurrence of toRemove
		std::string::size_type lastPos = str.find(toRemove, 0);
		if(lastPos > 0)
			result.append(str.substr(0, lastPos));

		// parse the rest
		while(lastPos != std::string::npos)
		{
			lastPos += toRemove.size();
			const std::string::size_type pos = str.find(toRemove, lastPos);
			result.append(str.substr(lastPos, pos - lastPos));
			lastPos = pos;
		}

		return result;
	}



	std::string Remove(const std::string& str, const std::vector<char>& toRemove)
	{
		std::string result;
		result.reserve(str.size());
		for(char c : str)
		{
			if(std::find(toRemove.begin(), toRemove.end(), c) == toRemove.end())
				result.push_back(c);
		}
		return result;
	}



	std::string Remove(const std::string& str, const std::vector<std::string>& toRemove)
	{
		std::string result = str;
		for(const std::string& s : toRemove)
			result = Remove(result, s);
		return result;
	}



	std::string Replace(const std::string& str, char from, char to)
	{
		std::string result;
		result.reserve(str.size());
		for(char c : str)
			result.push_back(c == from ? to : c);
		return result;
	}



	std::string Replace(const std::string& str, const std::string& from, const std::string& to)
	{
		std::string result = str;
		size_t pos = result.find(from);
		while(pos != std::string::npos)
		{
			result.replace(pos, from.size(), to);
			pos = result.find(from, pos + to.size());
		}
		return result;
	}




	std::string TimeString(int64_t elapsedNs)
	{
		if(elapsedNs < 1'000)
			return format("%lld ns", elapsedNs);
		if(elapsedNs < 1'000'000)
			return format("%lld.%01lld us", elapsedNs / 1'000, (elapsedNs / 100) % 10);
		if(elapsedNs < 1'000'000'000)
			return format("%lld.%01lld ms", elapsedNs / 1'000'000, (elapsedNs / 100'000) % 10);
		if(elapsedNs < 60'000'000'000)
			return format("%lld.%01lld s", elapsedNs / 1'000'000'000, (elapsedNs / 100'000'000) % 10);

		// (hh:)mm:ss format
		const int64_t t2 = elapsedNs / 1'000'000'000;
		if(t2 < 3600)
			return format("%02lld:%02lld", t2 / 60, t2 % 60);
		return format("%lld:%02lld:%02lld", t2 / 3600, (t2 % 3600) / 60, t2 % 60);
	}



	std::string ThousandSeparators(uint64_t val, const std::string& separator)
	{
		if(val < 1000)
			return std::to_string(val);

		std::string s = "";
		int i = 0;
		while(val > 0)
		{
			if(i && (i % 3) == 0)
				s = separator + s;
			s = "0123456789"[val % 10] + s;
			val /= 10;
			i++;
		}
		return s;
	}
}
