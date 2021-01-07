#include "Resource.h"

namespace Tracer
{
	bool Resource::IsDirty(bool parseDependencies) const
	{
		if(mIsDirty)
			return true;

		if(parseDependencies)
		{
			for(std::weak_ptr<Resource> d : mDependencies)
				if(!d.expired() && d.lock()->IsDirty())
					return true;
		}

		return false;
	}



	void Resource::AddDependency(std::shared_ptr<Resource> dependency)
	{
		if(!dependency)
			return;

		for(std::weak_ptr<Resource> w : mDependencies)
			if(!w.expired() && w.lock() == dependency)
				return;
		mDependencies.push_back(dependency);
	}



	void Resource::RemoveDependency(std::shared_ptr<Resource> dependency)
	{
		if(!dependency)
			return;

		for(std::vector<std::weak_ptr<Resource>>::iterator it = mDependencies.begin(); it != mDependencies.end(); it++)
		{
			if(!it->expired() && it->lock() == dependency)
			{
				mDependencies.erase(it);
				return;
			}
		}
	}
}
