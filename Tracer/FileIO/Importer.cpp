#include "Importer.h"

// Project
#include "Renderer/Scene.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
#include "Utility/Logger.h"
#include "Utility/Stopwatch.h"
#include "Utility/Utility.h"

// Assimp
#pragma warning(push)
#pragma warning(disable: 4061 4619 26451 26495)
#include "assimp/Importer.hpp"
#include "assimp/DefaultLogger.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#pragma warning(pop)

// FreeImage
#include "FreeImage/FreeImage.h"

// C++
#include <cassert>
#include <filesystem>
#include <stdexcept>
#include <map>

namespace Tracer
{
	namespace
	{
		// stream to attach to assimp logger
		class ImportLogStream : public Assimp::LogStream
		{
		public:
			void write(const char* message)
			{
				Logger::Info("[Assimp] %s", message);
			}
		};



		std::shared_ptr<Material> ImportMaterial(Scene* scene, const std::string& importDir, const aiMaterial* aMat)
		{
			// name
			aiString name;
			if(aMat->Get(AI_MATKEY_NAME, name) != AI_SUCCESS)
				return nullptr;

			// create material
			std::shared_ptr<Material> mat = std::make_shared<Material>(name.C_Str());

			// parse properties
			//ai_real r;
			aiColor3D c3;

			//if (!aMat->Get(AI_MATKEY_OPACITY, r))
			//	mat->SetOpacity(r);

			//if (!aMat->Get(AI_MATKEY_SHININESS, r))
			//	mat->SetShininess(r);

			//if (!aMat->Get(AI_MATKEY_REFRACTI, r))
			//	mat->SetRefractI(r);

			if (!aMat->Get(AI_MATKEY_COLOR_DIFFUSE, c3))
				mat->SetDiffuse(make_float3(c3.r, c3.g, c3.b));

			//if (!aMat->Get(AI_MATKEY_COLOR_SPECULAR, c3))
			//	mat->SetSpecular(make_float3(c3.r, c3.g, c3.b));

			if (!aMat->Get(AI_MATKEY_COLOR_EMISSIVE, c3))
				mat->SetEmissive(make_float3(c3.r, c3.g, c3.b));

			//if (!aMat->Get(AI_MATKEY_COLOR_TRANSPARENT, c3))
			//	mat->SetTransparent(make_float3(c3.r, c3.g, c3.b));

			// parse textures

			aiString texPath;
			if(aMat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == aiReturn_SUCCESS)
				mat->SetDiffuseMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_SPECULAR, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetSpecularMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_EMISSIVE, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetEmissiveMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_HEIGHT, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetHeightMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			if(aMat->GetTexture(aiTextureType_NORMALS, 0, &texPath) == aiReturn_SUCCESS)
				mat->SetNormalMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_SHININESS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetShininessMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_OPACITY, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetOpacityMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_DISPLACEMENT, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetDisplacementMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetBaseColorMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_EMISSION_COLOR, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetEmissionColorMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_METALNESS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetMetalnessMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetDiffuseRoughnessMap(Importer::ImportTexture(scene, texPath.C_Str(), importDir));

			return mat;
		}



		void ImportMesh(std::shared_ptr<Model> model, aiMesh* aMesh, const std::vector<std::shared_ptr<Material>>& materials)
		{
			// vertices
			std::vector<float3> positions(aMesh->mNumVertices, make_float3(0));
			std::vector<float3> normals(aMesh->mNumVertices, make_float3(0));
			std::vector<float2> texcoords(aMesh->mNumVertices, make_float2(0));

			for(unsigned int i = 0; i < aMesh->mNumVertices; i++)
			{
				if(aMesh->mVertices)
					positions[i] = *reinterpret_cast<float3*>(aMesh->mVertices + i);

				if(aMesh->mNormals)
					normals[i] = *reinterpret_cast<float3*>(aMesh->mNormals + i);

				if(aMesh->mTextureCoords[0])
					texcoords[i] = *reinterpret_cast<float2*>(aMesh->mTextureCoords[0] + i);
			}

			// indices
			std::vector<uint3> indices;
			indices.reserve(aMesh->mNumFaces);
			for(unsigned int i = 0; i < aMesh->mNumFaces; i++)
			{
				aiFace f = aMesh->mFaces[i];
				if(f.mNumIndices != 3)
					throw std::runtime_error("Encountered non-triangulated face during import");
				indices.push_back(make_uint3(f.mIndices[0], f.mIndices[1], f.mIndices[2]));
			}

			// add the mesh
			model->AddMesh(positions, normals, texcoords, indices, aMesh->mMaterialIndex);
		}



		//
		// cache read helpers
		//
		template<typename TYPE>
		TYPE Read(FILE* f)
		{
			TYPE val;
			fread(&val, sizeof(TYPE), 1, f);
			return val;
		}



		template<>
		std::string Read<std::string>(FILE* f)
		{
			size_t len;
			fread(&len, sizeof(size_t), 1, f);
			std::string s(len, '\0');
			fread(s.data(), sizeof(char), len, f);
			return s;
		}



		template<typename TYPE>
		std::vector<TYPE> ReadVec(FILE* f)
		{
			size_t elemCount;
			fread(&elemCount, sizeof(size_t), 1, f);
			std::vector<TYPE> v(elemCount);
			fread(v.data(), sizeof(TYPE), elemCount, f);
			return v;
		}



		//
		// cache write helpers
		//
		template<typename TYPE>
		void Write(FILE* f, const TYPE& data)
		{
			fwrite(&data, sizeof(TYPE), 1, f);
		}



		template<>
		void Write(FILE* f, const std::string& s)
		{
			size_t len = s.length();
			fwrite(&len, sizeof(size_t), 1, f);
			fwrite(s.data(), sizeof(char), len, f);
		}



		template<typename TYPE>
		void WriteVec(FILE* f, const std::vector<TYPE>& v)
		{
			size_t elemCount = v.size();
			fwrite(&elemCount, sizeof(size_t), 1, f);
			fwrite(v.data(), sizeof(TYPE), elemCount, f);
		}



		//
		// Model cache
		//
		bool SaveToCache(std::shared_ptr<Model> model, const std::string& filePath)
		{
			const size_t pathHash = std::hash<std::string>{}(filePath);
			const std::string cacheFile = "cache/" + std::to_string(pathHash);
			std::filesystem::create_directory("cache");
			FILE* f = nullptr;
			if((fopen_s(&f, cacheFile.c_str(), "wb") != 0) || !f)
				return false;

			// vertices
			WriteVec(f, model->Vertices());
			WriteVec(f, model->Normals());
			WriteVec(f, model->TexCoords());

			// indices
			WriteVec(f, model->Indices());
			WriteVec(f, model->MaterialIndices());

			// materials
			const auto& mats = model->Materials();
			Write(f, model->Materials().size());
			for(auto& m : mats)
			{
				Write(f, m->Name());
				Write(f, m->Diffuse());
				Write(f, m->Emissive());
				Write(f, m->DiffuseMap() ? m->DiffuseMap()->Path() : "");
				Write(f, m->NormalMap() ? m->NormalMap()->Path() : "");
			}

			// close the file
			fclose(f);

			return true;
		}



		std::shared_ptr<Model> LoadModelFromCache(Scene* scene, const std::string& filePath, const std::string& name = "")
		{
			const size_t pathHash = std::hash<std::string>{}(filePath);
			const std::string cacheFile = "cache/" + std::to_string(pathHash);
			FILE* f = nullptr;
			if((fopen_s(&f, cacheFile.c_str(), "rb") != 0) || !f)
				return nullptr;

			const std::string importDir = Directory(filePath);

			// vertices
			std::vector<float3> vertices = ReadVec<float3>(f);
			std::vector<float3> normals = ReadVec<float3>(f);
			std::vector<float2> texCoords = ReadVec<float2>(f);

			// indices
			std::vector<uint3> indices = ReadVec<uint3>(f);
			std::vector<uint32_t> materialIndices = ReadVec<uint32_t>(f);

			std::shared_ptr<Model> model = std::make_shared<Model>(filePath, name);
			model->Set(vertices, normals, texCoords, indices, materialIndices);

			// materials
			size_t matCount = 0;
			fread(&matCount, sizeof(size_t), 1, f);
			for(size_t i = 0; i < matCount; i++)
			{
				std::string matName = Read<std::string>(f);
				std::shared_ptr<Material> mat = std::make_shared<Material>(matName);

				mat->SetDiffuse(Read<float3>(f));
				mat->SetEmissive(Read<float3>(f));

				std::string diffMap = Read<std::string>(f);
				if(!diffMap.empty())
					mat->SetDiffuseMap(Importer::ImportTexture(scene, diffMap));

				std::string norMap = Read<std::string>(f);
				if(!norMap.empty())
					mat->SetNormalMap(Importer::ImportTexture(scene, norMap));

				model->AddMaterial(mat);
			}

			return model;
		}



		//
		// Texture cache
		//
		bool SaveToCache(std::shared_ptr<Texture> tex, const std::string& textureFile)
		{
			const size_t pathHash = std::hash<std::string>{}(textureFile);
			const std::string cacheFile = "cache/" + std::to_string(pathHash);
			std::filesystem::create_directory("cache");
			FILE* f = nullptr;
			if((fopen_s(&f, cacheFile.c_str(), "wb") != 0) || !f)
				return false;

			Write(f, tex->Resolution());
			WriteVec(f, tex->Pixels());
			fclose(f);
			return true;
		}



		std::shared_ptr<Texture> LoadTextureFromCache(Scene* scene, const std::string& filePath, const std::string& importDir = "")
		{
			const size_t pathHash = std::hash<std::string>{}(filePath);
			const std::string cacheFile = "cache/" + std::to_string(pathHash);
			FILE* f = nullptr;
			if((fopen_s(&f, cacheFile.c_str(), "rb") != 0) || !f)
				return nullptr;

			int2 res = Read<int2>(f);
			std::vector<float4> pixels = ReadVec<float4>(f);
			return std::make_shared<Texture>(filePath, res, pixels);
		}
	}



	const std::vector<Importer::Format>& Importer::SupportedModelFormats()
	{
		static std::vector<Format> result;
		if(result.size() == 0)
		{
			Assimp::Importer importer;
			aiString extensions;
			importer.GetExtensionList(extensions);

			const std::vector<std::string> exts = Split(extensions.C_Str(), ';');
			result.reserve(exts.size());
			for(const std::string& e : exts)
			{
				const std::string ext = e.substr(2);

				Format f;
				f.name = ext;
				f.description = ext;
				f.ext = ext;
				result.push_back(f);
			}
		}
		return result;
	}



	const std::vector<Importer::Format>& Importer::SupportedTextureFormats()
	{
		static std::vector<Format> result;
		if(result.size() == 0)
		{
			const int formatCount = FreeImage_GetFIFCount();
			result.reserve(formatCount);
			for(int i = 0; i < formatCount; i++)
			{
				FREE_IMAGE_FORMAT fif = static_cast<FREE_IMAGE_FORMAT>(i);
				if(FreeImage_FIFSupportsReading(fif))
				{
					Format f;
					f.name = FreeImage_GetFormatFromFIF(fif);
					f.description = FreeImage_GetFIFDescription(fif);
					f.ext = FreeImage_GetFIFExtensionList(fif);
					result.push_back(f);
				}
			}
		}
		return result;
	}



	std::shared_ptr<Texture> Importer::ImportTexture(Scene* scene, const std::string& filePath, const std::string& importDir)
	{
		std::string globalPath = filePath;
		if(!importDir.empty())
			globalPath = importDir + "/" + globalPath;
		globalPath = GlobalPath(globalPath);

		// check the scene for an existing texture
		if(std::shared_ptr<Texture> tex = scene->GetTexture(globalPath))
			return tex;

		// check the cache
		Stopwatch sw;
		std::shared_ptr<Texture> tex = LoadTextureFromCache(scene, globalPath, importDir);
		if(tex)
		{
			Logger::Info("Loaded \"%s\" from cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());
			return tex;
		}

		// load image
		FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(globalPath.c_str(), 0);
		if(fif == FIF_UNKNOWN)
			fif = FreeImage_GetFIFFromFilename(globalPath.c_str());

		FIBITMAP* tmp = FreeImage_Load(fif, globalPath.c_str());
		//FIBITMAP* dib = FreeImage_ConvertTo32Bits(tmp);
		FIBITMAP* dib = FreeImage_ConvertToRGBAF(tmp);
		FreeImage_Unload(tmp);

		const uint32_t width  = FreeImage_GetWidth(dib);
		const uint32_t height = FreeImage_GetHeight(dib);
		std::vector<float4> pixels(width * height);
		for(uint32_t y = 0; y < height; y++)
		{
			const uint8_t* line = FreeImage_GetScanLine(dib, y);
			memcpy(pixels.data() + (y * width), line, width * sizeof(float4));
		}
		FreeImage_Unload(dib);

		// create texture
		tex = std::make_shared<Texture>(globalPath, make_int2(width, height), pixels);
		Logger::Info("Imported \"%s\" in %s", globalPath.c_str(), sw.ElapsedString().c_str());

		sw.Reset();
		SaveToCache(tex, globalPath);
		Logger::Debug("Saved \"%s\" to cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());

		return tex;
	}



	std::shared_ptr<Model> Importer::ImportModel(Scene* scene, const std::string& filePath, const std::string& name)
	{
		Stopwatch sw;
		const std::string globalPath = GlobalPath(filePath);
		std::shared_ptr<Model> model = LoadModelFromCache(scene, globalPath, name);
		if(model)
		{
			Logger::Info("Loaded \"%s\"from cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());
			return model;
		}

		const std::string importDir = Directory(globalPath);

#if false
		// attach log stream
		static Assimp::Logger* defaultLogger = nullptr;
		if(!defaultLogger)
		{
			defaultLogger = Assimp::DefaultLogger::create();
			//defaultLogger->setLogSeverity(Assimp::Logger::VERBOSE);
			defaultLogger->attachStream(new ImportLogStream(), Assimp::Logger::Debugging | Assimp::Logger::Info | Assimp::Logger::Err | Assimp::Logger::Warn);
		}
#endif


		// Importer
		Assimp::Importer importer;
		if(!importer.IsExtensionSupported(FileExtension(globalPath)))
		{
			Logger::Error("Filetype \"%s\" is not supported by the importer.\n", FileExtension(globalPath).c_str());
			return nullptr;
		}

		// import flags
		constexpr uint32_t importFlags =
			//aiProcess_CalcTangentSpace |
			aiProcess_JoinIdenticalVertices |
			//aiProcess_MakeLeftHanded |
			aiProcess_Triangulate |
			aiProcess_RemoveComponent |
			//aiProcess_GenNormals |
			aiProcess_GenSmoothNormals |
			//aiProcess_SplitLargeMeshes |
			aiProcess_PreTransformVertices |
			aiProcess_LimitBoneWeights |
			aiProcess_ValidateDataStructure |
			//aiProcess_ImproveCacheLocality |
			aiProcess_RemoveRedundantMaterials |
			aiProcess_FixInfacingNormals |
			aiProcess_PopulateArmatureData |
			aiProcess_SortByPType |
			aiProcess_FindDegenerates |
			aiProcess_FindInvalidData |
			aiProcess_GenUVCoords |
			//aiProcess_TransformUVCoords |
			//aiProcess_FindInstances |
			aiProcess_OptimizeMeshes |
			//aiProcess_OptimizeGraph |
			//aiProcess_FlipUVs |
			//aiProcess_FlipWindingOrder |
			//aiProcess_SplitByBoneCount |
			//aiProcess_Debone |
			//aiProcess_GlobalScale |
			//aiProcess_EmbedTextures |
			//aiProcess_ForceGenNormals |
			//aiProcess_GenBoundingBoxes |
			0u;
		if(!importer.ValidateFlags(importFlags))
			throw std::runtime_error(importer.GetErrorString());

		// assimp properties
		importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
		importer.SetPropertyInteger(AI_CONFIG_FAVOUR_SPEED, 1);
		importer.SetPropertyInteger(AI_CONFIG_PP_LBW_MAX_WEIGHTS, 4);

		// read the file
		const aiScene* aScene = importer.ReadFile(globalPath.c_str(), importFlags);
		if(!aScene)
		{
			Logger::Error("Failed to import \"%s\": %s", globalPath.c_str(), importer.GetErrorString());
			return nullptr;
		}

		// create model
		model = std::make_shared<Model>(globalPath, name);

		// import materials
		std::map<std::string, std::shared_ptr<Texture>> textures;
		for(uint32_t i = 0; i < aScene->mNumMaterials; i++)
			model->AddMaterial(ImportMaterial(scene, importDir, aScene->mMaterials[i]));

		// import meshes
		uint32_t polyCount = 0;
		for(uint32_t i = 0; i < aScene->mNumMeshes; i++)
		{
			ImportMesh(model, aScene->mMeshes[i], model->Materials());
			polyCount += aScene->mMeshes[i]->mNumFaces;
		}

		Logger::Info("Imported \"%s\" in %s", globalPath.c_str(), sw.ElapsedString().c_str());
		Logger::Debug("  Meshes   : %s", ThousandSeparators(aScene->mNumMeshes).c_str());
		Logger::Debug("  Materials: %s", ThousandSeparators(model->Materials().size()).c_str());
		Logger::Debug("  Textures : %s", ThousandSeparators(textures.size()).c_str());
		Logger::Debug("  Polygons : %s", ThousandSeparators(polyCount).c_str());

		sw.Reset();
		SaveToCache(model, globalPath);
		Logger::Debug("Saved \"%s\" to cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());

		return model;
	}
}
