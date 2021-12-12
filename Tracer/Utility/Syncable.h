#pragma once

namespace Tracer
{
	class Syncable
	{
	public:
		Syncable() = default;
		virtual ~Syncable() {}

		virtual bool IsOutOfSync() const { return mOutOfSync; }
		virtual void MarkOutOfSync() { mOutOfSync = true; }
		virtual void MarkSynced() { mOutOfSync = false; }

	protected:
		bool mOutOfSync = true;
	};
}
