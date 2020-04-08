#pragma once

namespace Tracer
{
	class GuiWindow
	{
	public:
		virtual ~GuiWindow() {}
		bool IsEnabled() const { return mEnabled; }

		virtual void Draw() = 0;

	protected:
		bool mEnabled = true;
	};
}
