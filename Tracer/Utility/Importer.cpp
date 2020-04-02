#include "Importer.h"

// Project
#include "Resources/Material.h"
#include "Resources/Mesh.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
#include "Utility/Utility.h"

// Assimp
#pragma warning(push)
#pragma warning(disable: 4061 4619)
#include "assimp/cimport.h"
#include "assimp/config.h"
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
		std::shared_ptr<Texture> ImportTexture(const std::string& importDir, const std::string& textureFile)
		{
			const std::string texPath = importDir + "/" + textureFile;

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
			return std::make_shared<Texture>(FileName(textureFile), make_uint2(w, h), pixels);
		}



		std::shared_ptr<Material> ImportMaterial(const std::string& importDir, std::map<std::string, std::shared_ptr<Texture>>& textures, const aiMaterial* aMat)
		{
			// name
			aiString name;
			if(aMat->Get(AI_MATKEY_NAME, name) != AI_SUCCESS)
				return nullptr;

			// create material
			std::shared_ptr<Material> mat = std::make_shared<Material>(name.C_Str());

			// parse properties
			for(unsigned int i = 0; i < aMat->mNumProperties; i++)
			{
				const aiMaterialProperty* aProp = aMat->mProperties[i];

				// name
				std::string propName = aProp->mKey.C_Str();
				const size_t dotIx = propName.find_first_of('.');
				if(dotIx != std::string::npos)
					propName = propName.substr(dotIx + 1);

				// diffuse
				if(propName == "diffuse" && aProp->mDataLength == 3 * sizeof(float))
				{
					mat->mDiffuse = *reinterpret_cast<float3*>(aProp->mData);
					continue;
				}

				//printf("Unhandled material property: %s\n", propName.c_str());
			}

			// parse textures
			aiString texPath;
			if(aMat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == aiReturn_SUCCESS)
			{
				std::string texPathStr = texPath.C_Str();
				auto it = textures.find(texPathStr);
				if(it != textures.end())
					mat->mDiffuseMap = it->second;

				auto tex = ImportTexture(importDir, texPathStr);
				textures[texPathStr] = tex;
				mat->mDiffuseMap = tex;
			}

			return mat;

		}



		std::shared_ptr<Mesh> ImportMesh(aiMesh* aMesh, const std::vector<std::shared_ptr<Material>>& materials)
		{
			// vertices
			std::vector<float3> positions(aMesh->mNumVertices, make_float3(0, 0, 0));
			std::vector<float3> normals(aMesh->mNumVertices, make_float3(0, 0, 0));
			std::vector<float3> texcoords(aMesh->mNumVertices, make_float3(0, 0, 0));

			for(unsigned int i = 0; i < aMesh->mNumVertices; i++)
			{
				if(aMesh->mVertices)
					positions[i] = *reinterpret_cast<float3*>(aMesh->mVertices + i);

				if(aMesh->mNormals)
					normals[i] = *reinterpret_cast<float3*>(aMesh->mNormals + i);

				if(aMesh->mTextureCoords[0])
					texcoords[i] = *reinterpret_cast<float3*>(aMesh->mTextureCoords[0] + i);
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

			// create the mesh
			return std::make_shared<Mesh>(aMesh->mName.C_Str(), positions, normals, texcoords, indices, materials[aMesh->mMaterialIndex]);
		}
	}



	std::shared_ptr<Tracer::Model> ImportModel(const std::string& filePath)
	{
		const aiScene* aScene = nullptr;
		try
		{
			const std::string importDir = Directory(filePath);

			// Assimp flags
			const unsigned int importFlags =
				//aiProcess_CalcTangentSpace         |
				aiProcess_JoinIdenticalVertices    |
				//aiProcess_MakeLeftHanded           |
				aiProcess_Triangulate              |
				//aiProcess_RemoveComponent          |
				//aiProcess_GenNormals               |
				aiProcess_GenSmoothNormals         |
				//aiProcess_SplitLargeMeshes         |
				aiProcess_PreTransformVertices     |
				aiProcess_LimitBoneWeights         |
				aiProcess_ValidateDataStructure    |
				//aiProcess_ImproveCacheLocality     |
				aiProcess_RemoveRedundantMaterials |
				aiProcess_FixInfacingNormals       |
				aiProcess_SortByPType              |
				aiProcess_FindDegenerates          |
				aiProcess_FindInvalidData          |
				aiProcess_GenUVCoords              |
				//aiProcess_TransformUVCoords        |
				//aiProcess_FindInstances            |
				aiProcess_OptimizeMeshes           |
				//aiProcess_OptimizeGraph            |
				//aiProcess_FlipUVs                  |
				//aiProcess_FlipWindingOrder         |
				//aiProcess_SplitByBoneCount         |
				//aiProcess_Debone                   |
				0;

			// assimp properties
			aiPropertyStore* aProperties = aiCreatePropertyStore();
			aiSetImportPropertyInteger(aProperties, AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
			aiSetImportPropertyInteger(aProperties, AI_CONFIG_FAVOUR_SPEED, 1);
			aiSetImportPropertyInteger(aProperties, AI_CONFIG_PP_LBW_MAX_WEIGHTS, 4);

			// import the scene
			aScene = aiImportFileExWithProperties(filePath.c_str(), importFlags, nullptr, aProperties);
			if(!aScene)
				throw std::runtime_error("Assimp failed to parse the file");

			// create model
			std::shared_ptr<Model> model = std::make_shared<Model>(FileName(filePath));

			// import materials
			std::map<std::string, std::shared_ptr<Texture>> textures;
			for(uint32_t i = 0; i < aScene->mNumMaterials; i++)
				model->AddMaterial(ImportMaterial(importDir, textures, aScene->mMaterials[i]));

			// import meshes
			for(uint32_t i = 0; i < aScene->mNumMeshes; i++)
				model->AddMesh(ImportMesh(aScene->mMeshes[i], model->Materials()));

			// parse graph
			//ParseNode(...);

			aiReleaseImport(aScene);

			return model;
		}
		catch(const std::exception& e)
		{
			if(aScene)
				aiReleaseImport(aScene);

			printf("Failed to import \"%s\": %s\n", filePath.c_str(), e.what());
		}

		return nullptr;
	}
}
