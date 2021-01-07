#pragma once

namespace Tracer
{
	class GuiWindow
	{
	public:
		virtual ~GuiWindow() = default;

		template<class TYPE>
		static TYPE* const Get()
		{
			static TYPE inst = {};
			return &inst;
		}

		// drawing
		inline void SetEnabled(bool enable) { mEnabled = enable; }
		inline bool IsEnabled() const { return mEnabled; }
		inline void Draw()
		{
			if(mEnabled)
				DrawImpl();
		}

		// update
		virtual void Update() {}

	protected:
		// drawing
		virtual void DrawImpl() = 0;

		// drawing
		bool mEnabled = false;
	};
}
