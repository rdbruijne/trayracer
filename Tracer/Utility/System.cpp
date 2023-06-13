#include "System.h"

// C++
#include <filesystem>

// Windows
#include <ShellScalingApi.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>



namespace Tracer
{
	bool OpenFileDialog(const char* filter, const std::string& title, bool mustExist, std::string& result)
	{
		const std::string curDir = std::filesystem::current_path().string();

		char fileNameIn[MAX_PATH]  = {};
		char fileNameOut[MAX_PATH] = {};

		OPENFILENAMEA ofn = {};
		ofn.lStructSize    = sizeof(ofn);
		ofn.hwndOwner      = NULL;
		ofn.lpstrFilter    = filter;
		ofn.lpstrFile      = fileNameOut;
		ofn.nMaxFile       = MAX_PATH;
		ofn.lpstrFileTitle = fileNameIn;
		ofn.nMaxFileTitle  = MAX_PATH;
		ofn.lpstrTitle     = title.c_str();
		ofn.Flags          = OFN_EXPLORER | OFN_NONETWORKBUTTON | OFN_DONTADDTORECENT | (mustExist ? OFN_FILEMUSTEXIST : 0u);

		const bool ok = GetOpenFileNameA(&ofn);
		if(ok)
			result = fileNameOut;
		std::filesystem::current_path(curDir);
		return ok;
	}



	bool SaveFileDialog(const char* filter, const std::string& title, std::string& result)
	{
		const std::string curDir = std::filesystem::current_path().string();

		char fileNameIn[MAX_PATH]  = {};
		char fileNameOut[MAX_PATH] = {};

		OPENFILENAMEA ofn = {};
		ofn.lStructSize    = sizeof(ofn);
		ofn.hwndOwner      = NULL;
		ofn.lpstrFilter    = filter;
		ofn.lpstrFile      = fileNameOut;
		ofn.nMaxFile       = MAX_PATH;
		ofn.lpstrFileTitle = fileNameIn;
		ofn.nMaxFileTitle  = MAX_PATH;
		ofn.lpstrTitle     = title.c_str();
		ofn.Flags          = OFN_OVERWRITEPROMPT | OFN_EXPLORER | OFN_NONETWORKBUTTON | OFN_DONTADDTORECENT;

		const bool ok = GetSaveFileNameA(&ofn);
		if(ok)
			result = fileNameOut;
		std::filesystem::current_path(curDir);
		return ok;
	}



	float GetDisplayScale()
	{
		float x=1, y=1;
		GetDisplayScale(x, y);
		return x;
	}



	void GetDisplayScale(float& xScale, float yScale)
	{
		// Get the nearest mintor handle
		const HWND activeWindow = GetDesktopWindow();
		const HMONITOR monitor = MonitorFromWindow(activeWindow, MONITOR_DEFAULTTONEAREST);

		// Get the logical width and height of the monitor
		MONITORINFOEX monitorInfoEx;
		monitorInfoEx.cbSize = sizeof(monitorInfoEx);
		GetMonitorInfo(monitor, &monitorInfoEx);
		const long cxLogical = monitorInfoEx.rcMonitor.right - monitorInfoEx.rcMonitor.left;
		const long cyLogical = monitorInfoEx.rcMonitor.bottom - monitorInfoEx.rcMonitor.top;

		// Get the physical width and height of the monitor
		DEVMODE devMode;
		devMode.dmSize = sizeof(devMode);
		devMode.dmDriverExtra = 0;
		EnumDisplaySettings(monitorInfoEx.szDevice, ENUM_CURRENT_SETTINGS, &devMode);
		const DWORD cxPhysical = devMode.dmPelsWidth;
		const DWORD cyPhysical = devMode.dmPelsHeight;

		// Calculate the scaling factor
		const float cxScale = static_cast<float>(cxPhysical) / static_cast<float>(cxLogical);
		const float cyScale = static_cast<float>(cyPhysical) / static_cast<float>(cyLogical);

		// Return the result (rounded to 2 decimal places)
		xScale = roundf(cxScale * 100.f) / 100.f;
		yScale = roundf(cyScale * 100.f) / 100.f;
	}
}
