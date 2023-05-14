#pragma once

#include <string>

namespace Tracer
{
	// File dialog
	bool OpenFileDialog(const char* filter, const std::string& title, bool mustExist, std::string& result);
	bool SaveFileDialog(const char* filter, const std::string& title, std::string& result);

	// Display scale
	float GetDisplayScale();
	void GetDisplayScale(float& xScale, float yScale);
}
