#pragma once

// Project
#include "Utility/Defilable.h"
#include "Utility/Named.h"

// C++
#include <memory>
#include <string>
#include <vector>

namespace Tracer
{
	class Resource : public Defilable, public Named
	{
	public:
		Resource() = default;
		explicit Resource(const std::string& name) : Named(name) {}
		virtual ~Resource() {}

		// dirty
		bool IsDirty() const override { return IsDirty(true); }
		bool IsDirty(bool parseDependencies) const;

		// dependencies
		void AddDependency(std::shared_ptr<Resource> dependency);
		void RemoveDependency(std::shared_ptr<Resource> dependency);

	protected:
		std::vector<std::weak_ptr<Resource>> mDependencies;
	};
}
