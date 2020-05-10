#pragma once

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Resource
	{
	public:
		Resource() = default;
		explicit Resource(const std::string& name) : mName(name) {}
		virtual ~Resource() {}

		const std::string& Name() const { return mName; }

		bool IsDirty() const;
		inline void MarkDirty() { mIsDirty = true; }
		inline void MarkClean() { mIsDirty = false; }

		void AddDependency(std::shared_ptr<Resource> dependency);
		void RemoveDependency(std::shared_ptr<Resource> dependency);

	protected:
		std::string mName = "";
		bool mIsDirty = true;

		std::vector<std::weak_ptr<Resource>> mDependencies;
	};
}
