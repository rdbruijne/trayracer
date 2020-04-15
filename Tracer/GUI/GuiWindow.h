#pragma once

namespace Tracer
{
	class GuiWindow
	{
	public:
		virtual ~GuiWindow() = default;

		void Enable(bool enable = true) { mEnabled = enable; }
		bool IsEnabled() const { return mEnabled; }

		void Draw()
		{
			if(mEnabled)
				DrawImpl();
		}

	protected:
		virtual void DrawImpl() = 0;

		bool mEnabled = false;
	};
}
