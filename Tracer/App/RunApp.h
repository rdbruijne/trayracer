#pragma once

// C++
#include <string>

struct int2;
namespace Tracer
{
	class App;

	int RunApp(App* app, const std::string& windowTitle, const int2& resolution, bool fullscreen = false);
}
