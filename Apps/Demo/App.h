#pragma once

// Project
#include "ControlScheme.h"

// Tracer
#include "Tracer/App/App.h"

namespace Demo
{
	class App : public Tracer::App
	{
		// disable copying
		App(const App&) = delete;
		App& operator =(const App&) = delete;

		// disable moving
		App(App&&) = delete;
		App& operator =(const App&&) = delete;

	public:
		App() = default;

		void Init(Tracer::Renderer* renderer, Tracer::Window* window) override;
		void DeInit(Tracer::Renderer* renderer, Tracer::Window* window) override;
		void Tick(Tracer::Renderer* renderer, Tracer::Window* window, float dt) override;

	private:
		ControlScheme mControlScheme;
	};
}
