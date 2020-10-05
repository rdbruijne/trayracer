#pragma once

// Project
#include "ControlScheme.h"

// Tracer
#include "Tracer/App/App.h"

namespace Demo
{
	class App : public Tracer::App
	{
	public:
		App() = default;
		App(App&&) = delete;

		void Init(Tracer::Renderer* renderer, Tracer::Window* window) override;
		void DeInit(Tracer::Renderer* renderer, Tracer::Window* window) override;
		void Tick(Tracer::Renderer* renderer, Tracer::Window* window, float dt) override;

	private:
		ControlScheme mControlScheme;
	};
}
