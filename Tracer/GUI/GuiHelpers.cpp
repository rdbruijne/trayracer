#include "GuiHelpers.h"

// project
#include "OpenGL/Window.h"
#include "Utility/Utility.h"

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
		float gDpiScale = 1;
		ImGuiStyle gStyleBackup;



		void SetDpi(Window* window)
		{
			const float dpiScale = window->DpiScale();
			if(gDpiScale != dpiScale)
			{
				ImGuiStyle& style = ImGui::GetStyle();
				memcpy(&style, &gStyleBackup, sizeof(ImGuiStyle));
				style.ScaleAllSizes(dpiScale);

				ImGuiIO& io = ImGui::GetIO();
				io.FontGlobalScale = dpiScale;

				gDpiScale = dpiScale;
			}
		}
	}



	void GuiHelpers::Init(Window* renderWindow)
	{
		// init ImGUI
		IMGUI_CHECKVERSION();
		if(!ImGui::CreateContext())
			FatalError("Failed to create ImGUI context");

		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

		// style
		ImGui::StyleColorsDark();
		ImGuiStyle& style = ImGui::GetStyle();
		style.Colors[ImGuiCol_Header] = ImVec4(.5f, .5f, .75f, 1.f);
		memcpy(&gStyleBackup, &style, sizeof(ImGuiStyle));

		// DPI
		SetDpi(renderWindow);

		// init for OpenGL
		ImGui_ImplGlfw_InitForOpenGL(renderWindow->GlfwWindow(), true);
		ImGui_ImplOpenGL3_Init("#version 130");

		ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float);

		mRenderWindow = renderWindow;
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



	void GuiHelpers::DrawGui()
	{
		for(auto& [key, gui] : mGuiItems)
		{
			if(mRenderWindow->WasKeyPressed(key))
				gui->SetEnabled(!gui->IsEnabled());

			gui->Update();
			gui->Draw();
		}
	}
}
