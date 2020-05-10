#pragma once

// C++
#include <string>

namespace Tracer
{
	class Resource
	{
	public:
		Resource() = default;
		explicit Resource(const std::string& name) : mName(name) {}
		virtual ~Resource() {}

		const std::string& Name() const { return mName; }

		inline bool IsDirty() const { return mIsDirty; }
		inline void MarkDirty() { mIsDirty = true; }
		inline void MarkClean() { mIsDirty = false; }

	protected:
		std::string mName = "";
		bool mIsDirty = true;
	};
}
