#pragma once

namespace Tracer
{
	class Defilable
	{
	public:
		Defilable() = default;
		virtual ~Defilable() {}

		virtual bool IsDirty() const { return mIsDirty; }
		virtual void MarkDirty() { mIsDirty = true; }
		virtual void MarkClean() { mIsDirty = false; }

	protected:
		bool mIsDirty = true;
	};
}
