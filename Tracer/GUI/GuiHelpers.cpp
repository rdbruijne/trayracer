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

		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window->GlfwWindow(), true);
		ImGui_ImplOpenGL3_Init("#version 130");

		ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float);

		// #TODO(RJCDB): DPI Scaling
		//ImGuiStyle guiStyle;
		//guiStyle.ScaleAllSizes();
		//GetDpiForMonitor(nullptr, MDT_EFFECTIVE_DPI,

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
