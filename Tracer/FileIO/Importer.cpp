#include "Importer.h"

// Project
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
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
				printf("[ASSIMP] %s", message);
			}
		};



		std::shared_ptr<Material> ImportMaterial(const std::string& importDir, std::map<std::string, std::shared_ptr<Texture>>& textures, const aiMaterial* aMat)
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
			auto GetTex = [&](const char* texPath)
			{
				const std::string texPathStr = texPath;
				auto it = textures.find(texPathStr);
				if(it != textures.end())
					return it->second;

				auto tex = Importer::ImportTexture(texPathStr, importDir);
				textures[texPathStr] = tex;
				return tex;
			};

			aiString texPath;
			if(aMat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == aiReturn_SUCCESS)
				mat->SetDiffuseMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_SPECULAR, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetSpecularMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_EMISSIVE, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetEmissiveMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_HEIGHT, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetHeightMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_NORMALS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetNormalMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_SHININESS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetShininessMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_OPACITY, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetOpacityMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_DISPLACEMENT, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetDisplacementMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_DISPLACEMENT, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetBaseColorMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_EMISSION_COLOR, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetEmissionColorMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_METALNESS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetMetalnessMap(GetTex(texPath.C_Str()));

			//if(aMat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->SetDiffuseRoughnessMap(GetTex(texPath.C_Str()));

			return mat;
		}



		void ImportMesh(std::shared_ptr<Model> model, aiMesh* aMesh, const std::vector<std::shared_ptr<Material>>& materials)
		{
			// vertices
			std::vector<float3> positions(aMesh->mNumVertices, make_float3(0));
			std::vector<float3> normals(aMesh->mNumVertices, make_float3(0));
			std::vector<float3> tangents(aMesh->mNumVertices, make_float3(0));
			std::vector<float3> bitangents(aMesh->mNumVertices, make_float3(0));
			std::vector<float2> texcoords(aMesh->mNumVertices, make_float2(0));

			for(unsigned int i = 0; i < aMesh->mNumVertices; i++)
			{
				if(aMesh->mVertices)
					positions[i] = *reinterpret_cast<float3*>(aMesh->mVertices + i);

				if(aMesh->mNormals)
					normals[i] = *reinterpret_cast<float3*>(aMesh->mNormals + i);

				if(aMesh->mTangents)
					tangents[i] = *reinterpret_cast<float3*>(aMesh->mTangents + i);

				if(aMesh->mBitangents)
					bitangents[i] = *reinterpret_cast<float3*>(aMesh->mBitangents + i);

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
			model->AddMesh(positions, normals, tangents, bitangents, texcoords, indices, aMesh->mMaterialIndex);
		}
	}



	std::shared_ptr<Texture> Importer::ImportTexture(const std::string& textureFile, const std::string& importDir)
	{
		std::string texPath = textureFile;
		if(!importDir.empty())
			texPath = importDir + "/" + textureFile;

		// load image
		FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(texPath.c_str(), 0);
		if(fif == FIF_UNKNOWN)
			fif = FreeImage_GetFIFFromFilename(texPath.c_str());
		FIBITMAP* tmp = FreeImage_Load(fif, texPath.c_str());
		FIBITMAP* dib = FreeImage_ConvertTo32Bits(tmp);
		FreeImage_Unload(tmp);
		const uint32_t w = FreeImage_GetWidth(dib);
		const uint32_t h = FreeImage_GetHeight(dib);
		std::vector<uint32_t> pixels(static_cast<size_t>(w) * h);
		for(uint32_t y = 0; y < h; y++)
		{
			const uint8_t* line = FreeImage_GetScanLine(dib, y);
			memcpy(pixels.data() + (static_cast<size_t>(y) * w), line, w * sizeof(uint32_t));
		}
		FreeImage_Unload(dib);

		// create texture
		return std::make_shared<Texture>(texPath, make_uint2(w, h), pixels);
	}



	std::shared_ptr<Model> Importer::ImportModel(const std::string& filePath, const std::string& name)
	{
		printf("Importing \"%s\"\n", filePath.c_str());
		try
		{
			const std::string importDir = Directory(filePath);

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

			// import flags
			constexpr uint32_t importFlags =
				aiProcess_CalcTangentSpace |
				aiProcess_JoinIdenticalVertices |
				//aiProcess_MakeLeftHanded |
				aiProcess_Triangulate |
				//aiProcess_RemoveComponent |
				//aiProcess_GenNormals |
				aiProcess_GenSmoothNormals |
				//aiProcess_SplitLargeMeshes |
				aiProcess_PreTransformVertices |
				aiProcess_LimitBoneWeights |
				aiProcess_ValidateDataStructure |
				aiProcess_ImproveCacheLocality |
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
				aiProcess_GenBoundingBoxes |
				0u;
			if(!importer.ValidateFlags(importFlags))
				throw std::runtime_error(importer.GetErrorString());

			// assimp properties
			importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
			importer.SetPropertyInteger(AI_CONFIG_FAVOUR_SPEED, 1);
			importer.SetPropertyInteger(AI_CONFIG_PP_LBW_MAX_WEIGHTS, 4);

			// read the file
			const aiScene* aScene = importer.ReadFile(filePath.c_str(), importFlags);
			if(!aScene)
				throw std::runtime_error(importer.GetErrorString());

			// create model
			std::shared_ptr<Model> model = std::make_shared<Model>(filePath, name);

			// import materials
			std::map<std::string, std::shared_ptr<Texture>> textures;
			for(uint32_t i = 0; i < aScene->mNumMaterials; i++)
				model->AddMaterial(ImportMaterial(importDir, textures, aScene->mMaterials[i]));

			// import meshes
			uint32_t polyCount = 0;
			for(uint32_t i = 0; i < aScene->mNumMeshes; i++)
			{
				ImportMesh(model, aScene->mMeshes[i], model->Materials());
				polyCount += aScene->mMeshes[i]->mNumFaces;
			}

			// parse graph
			//ParseNode(...);

			printf("Imported \"%s\":\n", filePath.c_str());
			printf("  Meshes   : %s\n", ThousandSeparators(aScene->mNumMeshes).c_str());
			printf("  Materials: %s\n", ThousandSeparators(model->Materials().size()).c_str());
			printf("  Textures : %s\n", ThousandSeparators(textures.size()).c_str());
			printf("  Polygons : %s\n", ThousandSeparators(polyCount).c_str());

			return model;
		}
		catch(const std::exception& e)
		{
			printf("Failed to import \"%s\": %s\n", filePath.c_str(), e.what());
		}

		return nullptr;
	}
}
