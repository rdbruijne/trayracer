#include "RecorderGui.h"

// Project
#include "CameraPath.h"

// Tracer
#include "Tracer/GUI/GuiHelpers.h"
#include "Tracer/Resources/CameraNode.h"
#include "Tracer/Utility/Strings.h"

// ImGUI
#include "imgui/imgui.h"

// C++
#include <functional>

using Tracer::format;

namespace Benchmark
{
	namespace
	{
		void DisplayError(const std::string& error)
		{
			ImGui::Text(error.c_str());
			ImGui::End();
		}



		inline void TableHeader(const std::string& text)
		{
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::Text(text.c_str());
		}



		template<int Indent=0, typename ... Arg>
		inline void TableRow(const std::string& identifier, const std::string& fmt, Arg... args)
		{
			constexpr float IndentSize = 16.f;

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);

			const float indent = IndentSize * (Indent + 1);
			ImGui::Indent(indent);
			ImGui::Text(identifier.c_str());
			ImGui::Unindent(indent);
			ImGui::TableNextColumn();

			ImGui::Text(format(fmt.c_str(), std::forward<Arg>(args)...).c_str());
		}
	}



	void RecorderGui::DrawImpl()
	{
		// reset
		mStartPlayback = false;
		mStopPlayback = false;

		// window
		if(!ImGui::Begin("Recorder", &mEnabled))
		{
			ImGui::End();
			return;
		}

		// check for members
		if(!mPath)
		{
			DisplayError("No path assigned.");
			return;
		}

		Tracer::CameraNode* cameraNode = Tracer::GuiHelpers::GetCamNode();
		if(!cameraNode)
		{
			DisplayError("No camera node detected");
			return;
		}

		// playback
		if(mPlayTime < 0)
		{
			mStartPlayback = ImGui::Button("Play");
		}
		else
		{
			mStopPlayback = ImGui::Button("Stop");
			ImGui::SameLine();
			ImGui::Text("%.2f", mPlayTime);
		}

		// benchmark data
		ImGui::Separator();
		if(mBenchFrameCount == 0)
		{
			ImGui::Text("Waiting for benchmark results...");
		}
		else if(ImGui::BeginTable("Benchmark data", 2, ImGuiTableFlags_RowBg))
		{
			auto PerSec = [](uint64_t count, float ms) -> float { return static_cast<float>(count) / (ms * 1e-3f); };

			const Tracer::RenderStatistics::DeviceStatistics& deviceStats = mBenchStats.device;

			TableHeader("Framerate");
			TableRow("Frame count", "%.0f", mBenchFrameCount);
			TableRow("FPS", "%.1f", static_cast<float>(mBenchFrameCount * 1e3f) / mBenchTimeMs);
			TableRow("Frame time", "%.2f ms", mBenchTimeMs / mBenchFrameCount);

			TableHeader("Ray times");
			TableRow("Primary rays", "%.1f ms", deviceStats.primaryPathTimeMs / mBenchFrameCount);
			TableRow("Secondary rays", "%.1f ms", deviceStats.secondaryPathTimeMs / mBenchFrameCount);
			TableRow("Deep rays", "%.1f ms", deviceStats.deepPathTimeMs / mBenchFrameCount);
			TableRow("Shadow rays", "%.1f ms", deviceStats.shadowTimeMs / mBenchFrameCount);

			TableHeader("Rays/sec");
			TableRow("Rays", "%.1f M/s", PerSec(deviceStats.pathCount, mBenchTimeMs) / 1e6f);
			TableRow("Primaries", "%.1f M/s", PerSec(deviceStats.primaryPathCount, deviceStats.primaryPathTimeMs) / 1e6f);
			TableRow("Secondaries", "%.1f M/s", PerSec(deviceStats.secondaryPathCount, deviceStats.secondaryPathTimeMs) / 1e6f);
			TableRow("Deep", "%.1f M/s", PerSec(deviceStats.deepPathCount, deviceStats.deepPathTimeMs) / 1e6f);
			TableRow("Shadow", "%.1f M/s", PerSec(deviceStats.shadowRayCount, deviceStats.shadowTimeMs) / 1e6f);

			TableHeader("Post");
			TableRow("Shade time", "%.1f ms", deviceStats.shadeTimeMs / mBenchFrameCount);
			TableRow("Denoise time", "%.1f ms", mBenchStats.denoiseTimeMs / mBenchFrameCount);

			ImGui::EndTable();
		}

		// recorder
		ImGui::Separator();
		if(ImGui::CollapsingHeader("Recorder"))
		{
			// save / load / clear
			if(ImGui::Button("Save"))
				mPath->Save(PathFile);

			ImGui::SameLine();
			if(ImGui::Button("Load"))
				mPath->Load(PathFile);

			ImGui::SameLine();
			if(ImGui::Button("Clear"))
				mPath->SetNodes({});

			// record position
			ImGui::NewLine();
			ImGui::SliderFloat("Time", &mTime, 0.f, 10.f);
			if(ImGui::Button("Add transform"))
				mPath->Add(*cameraNode, mTime);

			// list nodes
			ImGui::NewLine();
			ImGui::Separator();
			ImGui::Text("Nodes");
			if(ImGui::BeginTable("Nodes", 2, ImGuiTableFlags_RowBg))
			{
				std::vector<Tracer::CameraNode>& nodes = mPath->Nodes();
				int i = 0;
				for (Tracer::CameraNode& n : nodes)
				{
					float t = __uint_as_float(n.Flags());

					ImGui::TableNextRow();

					// column 0
					ImGui::TableNextColumn();
					ImGui::Text("%2i: %.2f (%.2f %.2f %.2f)", i++, t, n.Position().x, n.Position().y, n.Position().z);

					// column 1
					ImGui::TableNextColumn();
					if(ImGui::Button("Select"))
					{
						mTime = __uint_as_float(n.Flags());
						*cameraNode = n;
					}
					ImGui::SameLine();
					if(ImGui::Button("Set"))
					{
						n = *cameraNode;
						n.SetFlags(__float_as_uint(mTime));
					}
					//if(ImGui::Button("Remove"))
				}
				ImGui::EndTable();
			}
		}

		ImGui::End();
	}
}
