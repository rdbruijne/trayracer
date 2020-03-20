#include "GuiHelpers.h"

// project
#include "OpenGL/Window.h"

// GUI
#include "GUI/CameraWindow.h"
#include "GUI/DebugWindow.h"
#include "GUI/RendererWindow.h"

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
		ImGui_ImplGlfw_InitForOpenGL(window->GetGlfwWindow(), true);
		ImGui_ImplOpenGL3_Init("#version 130");

		// #TODO(RJCDB): DPI Scaling
		//ImGuiStyle guiStyle;
		//guiStyle.ScaleAllSizes();
		//GetDpiForMonitor(nullptr, MDT_EFFECTIVE_DPI,

		// register child windows
		msWindows.push_back(new CameraWindow());
		msWindows.push_back(new DebugWindow());
		msWindows.push_back(new RendererWindow());

		return true;
	}



	void GuiHelpers::DeInit()
	{
		// destroy child windows
		for(GuiWindow* w : msWindows)
			delete w;
		msWindows.clear();

		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}



	void GuiHelpers::Draw()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// draw child windows
		for(GuiWindow* w : msWindows)
		{
			//if(w->IsEnabled())
				w->Draw();
		}

		ImGui::Render();

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

}
