#include "GuiHelpers.h"

// project
#include "OpenGL/Window.h"

// ImGUI
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

// GL
#include "glfw/glfw3.h"

namespace Tracer
{
	namespace
	{
		float sDpiScale = 1;
		ImGuiStyle sStyleBackup;



		void SetDpi(Window* window)
		{
			const float dpiScale = Window::MonitorDPI(window->CurrentMonitor());
			if(sDpiScale != dpiScale)
			{
				ImGuiStyle& style = ImGui::GetStyle();
				memcpy(&style, &sStyleBackup, sizeof(ImGuiStyle));
				style.ScaleAllSizes(dpiScale);

				ImGuiIO& io = ImGui::GetIO();
				io.FontGlobalScale = dpiScale;
			}
		}
	}



	bool GuiHelpers::Init(Window* window)
	{
		// init ImGUI
		IMGUI_CHECKVERSION();
		if(!ImGui::CreateContext())
		{
			printf("Failed to create ImGUI context\n");
			return false;
		}

		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

		// style
		ImGui::StyleColorsDark();
		ImGuiStyle& style = ImGui::GetStyle();
		memcpy(&sStyleBackup, &style, sizeof(ImGuiStyle));

		// DPI
		SetDpi(window);

		// init for OpenGL
		ImGui_ImplGlfw_InitForOpenGL(window->GlfwWindow(), true);
		ImGui_ImplOpenGL3_Init("#version 130");

		ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float);

		return true;
	}



	void GuiHelpers::DeInit()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}



	void GuiHelpers::BeginFrame()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}



	void GuiHelpers::EndFrame()
	{
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}
