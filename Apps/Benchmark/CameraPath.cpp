#include "CameraPath.h"

// Tracer
#include "Tracer/FileIO/JsonHelpers.h"
#include "Tracer/Logging/Logger.h"
#include "Tracer/Utility/FileSystem.h"
#include "Tracer/Utility/Stopwatch.h"
#include "Tracer/Utility/Strings.h"

// C++
#include <fstream>

using namespace rapidjson;

// Keys
#define Key_AutoFocus          "autofocus"
#define Key_Aperture           "aperture"
#define Key_BokehRotation      "bokehrotation"
#define Key_BokehSideCount     "bokehsides"
#define Key_Camera             "camera"
#define Key_Distortion         "distortion"
#define Key_FocalDist          "focaldist"
#define Key_Fov                "fov"
#define Key_Nodes              "nodes"
#define Key_Position           "position"
#define Key_Target             "target"
#define Key_Time               "time"
#define Key_Up                 "up"

namespace Benchmark
{
	float CameraPath::TotalTime() const
	{
		float totalTime = 0;
		for(const Tracer::CameraNode& node : mNodes)
		{
			const float t = __uint_as_float(node.Flags());
			totalTime += t;
		}
		return totalTime;
	}



	std::optional<Tracer::CameraNode> CameraPath::Playback(float time)
	{
		if(time < 0)
			return std::nullopt;

		// find start node
		Tracer::CameraNode* from = nullptr;
		Tracer::CameraNode* to = nullptr;

		float t = 0;
		for(size_t i = 0, j = 1; j < mNodes.size(); i++, j++)
		{
			const float t1 = __uint_as_float(mNodes[j].Flags());
			if(t1 > time)
			{
				t = time / t1;
				from = &mNodes[i];
				to = &mNodes[j];
				break;
			}
			time -= t1;
		}

		if(!from)
		{
			return std::nullopt;
		}

		// interpolate
		Tracer::CameraNode node;
		node.SetTransform(lerp(from->Transform(), to->Transform(), t));
		node.SetAperture(lerp(from->Aperture(), to->Aperture(), t));
		node.SetDistortion(lerp(from->Distortion(), to->Distortion(), t));
		node.SetFocalDist(lerp(from->FocalDist(), to->FocalDist(), t));
		node.SetFov(lerp(from->Fov(), to->Fov(), t));
		node.SetBokehSideCount(static_cast<int>(lerp(static_cast<float>(from->BokehSideCount()), static_cast<float>(to->BokehSideCount()), t)));
		node.SetBokehRotation(lerp(from->BokehRotation(), to->BokehRotation(), t));
		return node;
	}



	bool CameraPath::Load(const std::string& filepath)
	{
		Tracer::Logger::Info("Loading scene from \"%s\"", filepath.c_str());
		Tracer::Stopwatch sw;

		// read file from disk
		Document doc;
		{
			std::ifstream f(filepath);
			if(!f.is_open())
			{
				Tracer::Logger::Error("Failed to open \"%s\" for reading.", filepath.c_str());
				return false;
			}

			IStreamWrapper isw(f);
			doc.ParseStream(isw);
			f.close();
			if(!doc.IsObject())
			{
				Tracer::Logger::Error("\"%s\" does not contain valid json code.", filepath.c_str());
				return false;
			}
		}

		// clear nodes
		mNodes.clear();

		// load new nodes
		if(doc.HasMember(Key_Nodes))
		{
			const Value& jsonNodeList = doc[Key_Nodes];
			if(jsonNodeList.IsArray())
			{
				for(SizeType nodeIx = 0; nodeIx < jsonNodeList.Size(); nodeIx++)
				{
					const Value& jsonNode = jsonNodeList[nodeIx];

					float3 pos = make_float3(0, 0, -1);
					float3 target = make_float3(0, 0, 0);
					float3 up = make_float3(0, 1, 0);
					float aperture = 0;
					float distortion = 0;
					float focalDist = 1e5f;
					float fov = 90.f;
					float bokehRotation = 0;
					int bokehSideCount = 0;
					float time = 0;

					Tracer::Read(jsonNode, Key_Position, pos);
					Tracer::Read(jsonNode, Key_Target, target);
					Tracer::Read(jsonNode, Key_Up, up);
					Tracer::Read(jsonNode, Key_Aperture, aperture);
					Tracer::Read(jsonNode, Key_Distortion, distortion);
					Tracer::Read(jsonNode, Key_FocalDist, focalDist);
					Tracer::Read(jsonNode, Key_Fov, fov);
					Tracer::Read(jsonNode, Key_BokehRotation, bokehRotation);
					Tracer::Read(jsonNode, Key_BokehSideCount, bokehSideCount);
					Tracer::Read(jsonNode, Key_Time, time);

					// corrections
					fov = fminf(fmaxf(fov, .1f), 179.9f);

					if(pos == target)
						target = pos + make_float3(0, 0, 1);

					// set camera
					Tracer::CameraNode camNode = Tracer::CameraNode(pos, target, up, fov * DegToRad);
					camNode.SetAperture(aperture);
					camNode.SetDistortion(distortion);
					camNode.SetFocalDist(focalDist);
					camNode.SetBokehRotation(bokehRotation);
					camNode.SetBokehSideCount(bokehSideCount);
					camNode.SetFlags(__float_as_uint(time));
					mNodes.push_back(camNode);
				}
			}
		}

		Tracer::Logger::Info("Loaded camera path in %s", sw.ElapsedString().c_str());
		return true;
	}



	bool CameraPath::Save(const std::string& filepath)
	{
		std::string globalPath = Tracer::GlobalPath(filepath);
		if(Tracer::ToLower(Tracer::FileExtension(globalPath)) != ".path")
			globalPath += ".path";

		Tracer::Logger::Info("Saving camera path to \"%s\"", globalPath.c_str());
		Tracer::Stopwatch sw;

		// create json document
		Document doc;
		Document::AllocatorType& allocator = doc.GetAllocator();
		doc.SetObject();

		// export nodes
		Value nodeList = Value(kArrayType);
		for(const Tracer::CameraNode& node : mNodes)
		{
			Value jsonCam = Value(kObjectType);
			Tracer::Write(jsonCam, allocator, Key_Position, node.Position());
			Tracer::Write(jsonCam, allocator, Key_Target, node.Target());
			Tracer::Write(jsonCam, allocator, Key_Up, node.Up());
			Tracer::Write(jsonCam, allocator, Key_Aperture, node.Aperture());
			Tracer::Write(jsonCam, allocator, Key_Distortion, node.Distortion());
			Tracer::Write(jsonCam, allocator, Key_FocalDist, node.FocalDist());
			Tracer::Write(jsonCam, allocator, Key_Fov, node.Fov() * RadToDeg);
			Tracer::Write(jsonCam, allocator, Key_BokehRotation, node.BokehSideCount());
			Tracer::Write(jsonCam, allocator, Key_BokehSideCount, node.BokehSideCount());
			Tracer::Write(jsonCam, allocator, Key_Time, __uint_as_float(node.Flags()));
			nodeList.PushBack(jsonCam, allocator);
		}
		doc.AddMember(Key_Nodes, nodeList, allocator);

		// write to disk
		std::ofstream f(globalPath);
		if(!f.is_open())
		{
			Tracer::Logger::Error("Failed to open \"%s\" for writing.", filepath.c_str());
			return false;
		}

		OStreamWrapper osw(f);
		PrettyWriter<OStreamWrapper> writer(osw);
		writer.SetFormatOptions(PrettyFormatOptions::kFormatSingleLineArray);
		if(!doc.Accept(writer))
		{
			Tracer::Logger::Error("Failed to write camera path to \"%s\".", filepath.c_str());
			return false;
		}

		Tracer::Logger::Info("Saved camera path in %s", sw.ElapsedString().c_str());
		return true;
	}
}
