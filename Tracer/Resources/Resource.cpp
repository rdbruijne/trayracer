#include "Resource.h"

namespace Tracer
{
	bool Resource::IsDirty() const
	{
		if(mIsDirty)
			return true;

		for(auto& d : mDependencies)
			if(!d.expired() && d.lock()->IsDirty())
				return true;

		return false;
	}



	void Resource::AddDependency(std::shared_ptr<Resource> dependency)
	{
		for(auto& w : mDependencies)
			if(!w.expired() && w.lock() == dependency)
				return;
		mDependencies.push_back(dependency);
	}



	void Resource::RemoveDependency(std::shared_ptr<Resource> dependency)
	{
		for(auto it = mDependencies.begin(); it != mDependencies.end(); it++)
		{
			if(!it->expired() && it->lock() == dependency)
			{
				mDependencies.erase(it);
				return;
			}
		}
	}
}
