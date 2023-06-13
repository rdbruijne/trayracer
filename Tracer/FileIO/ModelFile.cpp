#include "ModelFile.h"

// Project
#include "FileIO/BinaryFile.h"
#include "FileIO/TextureFile.h"
#include "Renderer/Scene.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
#include "Utility/FileSystem.h"
#include "Utility/Logger.h"
#include "Utility/Stopwatch.h"
#include "Utility/Strings.h"

// Assimp
#include "assimp/Importer.hpp"
#include "assimp/DefaultLogger.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

// C++
#include <filesystem>

#define MODEL_CACHE_ENABLED		true
#define MODEL_CHACHE_ID_0		'M'
#define MODEL_CHACHE_ID_1		'D'
#define MODEL_CHACHE_ID_2		'L'
#define MODEL_CACHE_ID			{ MODEL_CHACHE_ID_0, MODEL_CHACHE_ID_1, MODEL_CHACHE_ID_2 }
#define MODEL_CACHE_VERSION		2



namespace Tracer
{
	namespace
	{
		// stream to attach to assimp logger
		template<Logger::Severity Severity>
		class ImportLogStream : public Assimp::LogStream
		{
		public:
			void write(const char* message) override
			{
				Logger::Message(Severity, "[Assimp] %s", message);
			}
		};



		//
		// Model cache
		//
		bool SaveToCache(std::shared_ptr<Model> model, const std::string& filePath)
		{
			// create file
			const std::string cacheFile = BinaryFile::CachedFilePath(filePath);
			BinaryFile f(cacheFile, BinaryFile::FileMode::Write);

			// write the header
			const BinaryFile::Header header = { MODEL_CACHE_ID, MODEL_CACHE_VERSION };
			f.Write(header);

			// vertices
			f.WriteVec(model->Vertices());
			f.WriteVec(model->Normals());
			f.WriteVec(model->TexCoords());

			// indices
			f.WriteVec(model->Indices());
			f.WriteVec(model->MaterialIndices());

			// materials
			const std::vector<std::shared_ptr<Material>>& materials = model->Materials();
			f.Write(model->Materials().size());
			for(const std::shared_ptr<Material>& mat : materials)
			{
				f.Write(mat->Name());
				for(size_t i = 0; i < magic_enum::enum_count<MaterialPropertyIds>(); i++)
				{
					const MaterialPropertyIds id = static_cast<MaterialPropertyIds>(i);
					if(mat->IsFloatColorEnabled(id))
						f.Write(mat->FloatColor(id));
					if(mat->IsRgbColorEnabled(id))
						f.Write(mat->RgbColor(id));
					if(mat->IsTextureEnabled(id))
						f.Write(mat->TextureMap(id) ? mat->TextureMap(id)->Path() : "");
				}
			}

			f.Flush();

			return true;
		}



		std::shared_ptr<Model> LoadFromCache(Scene* scene, const std::string& filePath, const std::string& name = "")
		{
			// cache file name
			const std::string cacheFile = BinaryFile::CachedFilePath(filePath);

			// compare write times
			if(!FileExists(cacheFile) || FileLastWriteTime(filePath) > FileLastWriteTime(cacheFile))
				return nullptr;

			// open cache file
			BinaryFile f(cacheFile, BinaryFile::FileMode::Read);

			// check the header
			BinaryFile::Header header = f.Read<BinaryFile::Header>();
			if((header.type[0] != MODEL_CHACHE_ID_0) || (header.type[1] != MODEL_CHACHE_ID_1) ||
			   (header.type[2] != MODEL_CHACHE_ID_2) || (header.version != MODEL_CACHE_VERSION))
			{
				Logger::Error("Invalid cache file for \"%s\" (%s): found %c%c%c %d, expected %c%c%c %d",
							  filePath.c_str(), cacheFile.c_str(),
							  header.type[0], header.type[1], header.type[2], header.version,
							  MODEL_CHACHE_ID_0, MODEL_CHACHE_ID_1, MODEL_CHACHE_ID_2, MODEL_CACHE_VERSION);
				return nullptr;
			}

			// vertices
			const std::vector<float3> vertices  = f.ReadVec<float3>();
			const std::vector<float3> normals   = f.ReadVec<float3>();
			const std::vector<float2> texCoords = f.ReadVec<float2>();

			// indices
			const std::vector<uint3> indices = f.ReadVec<uint3>();
			const std::vector<uint32_t> materialIndices = f.ReadVec<uint32_t>();

			// initialize the model
			std::shared_ptr<Model> model = std::make_shared<Model>(filePath, name);
			model->Set(vertices, normals, texCoords, indices, materialIndices);

			// materials
			size_t matCount = f.Read<size_t>();
			for(size_t matIx = 0; matIx < matCount; matIx++)
			{
				const std::string matName = f.Read<std::string>();
				std::shared_ptr<Material> mat = std::make_shared<Material>(matName);

				for(size_t propIx = 0; propIx < magic_enum::enum_count<MaterialPropertyIds>(); propIx++)
				{
					const MaterialPropertyIds id = static_cast<MaterialPropertyIds>(propIx);

					if(mat->IsFloatColorEnabled(id))
						mat->Set(id, f.Read<float>());

					if(mat->IsRgbColorEnabled(id))
						mat->Set(id, f.Read<float3>());

					if(mat->IsTextureEnabled(id))
					{
						const std::string texPath = f.Read<std::string>();
						if(!texPath.empty())
							mat->Set(id, TextureFile::Import(scene, texPath));
					}
				}

				model->AddMaterial(mat);
			}

			return model;
		}



		//
		// import
		//
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
			//	mat->Set(MaterialPropertyIds::Opacity, make_float3(r));

			//if (!aMat->Get(AI_MATKEY_SHININESS, r))
			//	mat->Set(MaterialPropertyIds::Shininess, make_float3(r));

			//if (!aMat->Get(AI_MATKEY_REFRACTI, r))
			//	mat->Set(MaterialPropertyIds::RefractI, make_float3(r));

			if (!aMat->Get(AI_MATKEY_COLOR_DIFFUSE, c3))
				mat->Set(MaterialPropertyIds::Diffuse, make_float3(c3.r, c3.g, c3.b));

			//if (!aMat->Get(AI_MATKEY_COLOR_SPECULAR, c3))
			//	mat->Set(MaterialPropertyIds::Specular, make_float3(c3.r, c3.g, c3.b));

			if (!aMat->Get(AI_MATKEY_COLOR_EMISSIVE, c3))
				mat->Set(MaterialPropertyIds::Emissive, make_float3(c3.r, c3.g, c3.b));

			//if (!aMat->Get(AI_MATKEY_COLOR_TRANSPARENT, c3))
			//	mat->Set(MaterialPropertyIds::Transparent, make_float3(c3.r, c3.g, c3.b));

			// parse textures
			aiString texPath;
			if(aMat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == aiReturn_SUCCESS)
				mat->Set(MaterialPropertyIds::Diffuse, TextureFile::Import(scene, texPath.C_Str(), importDir));

			if(aMat->GetTexture(aiTextureType_SPECULAR, 0, &texPath) == aiReturn_SUCCESS)
				mat->Set(MaterialPropertyIds::Specular, TextureFile::Import(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_EMISSIVE, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->Set(MaterialPropertyIds::Emissive, TextureFile::Import(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_HEIGHT, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->Set(MaterialPropertyIds::Height, TextureFile::Import(scene, texPath.C_Str(), importDir));

			if(aMat->GetTexture(aiTextureType_NORMALS, 0, &texPath) == aiReturn_SUCCESS)
				mat->Set(MaterialPropertyIds::Normal, TextureFile::Import(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_SHININESS, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->Set(MaterialPropertyIds::Shininess, TextureFile::Import(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_OPACITY, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->Set(MaterialPropertyIds::Opacity, TextureFile::Import(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_DISPLACEMENT, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->Set(MaterialPropertyIds::Displacement, TextureFile::Import(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->Set(MaterialPropertyIds::BaseColor, TextureFile::Import(scene, texPath.C_Str(), importDir));

			//if(aMat->GetTexture(aiTextureType_EMISSION_COLOR, 0, &texPath) == aiReturn_SUCCESS)
			//	mat->Set(MaterialPropertyIds::EmissionColor, TextureFile::Import(scene, texPath.C_Str(), importDir));

			if(aMat->GetTexture(aiTextureType_METALNESS, 0, &texPath) == aiReturn_SUCCESS)
				mat->Set(MaterialPropertyIds::Metallic, TextureFile::Import(scene, texPath.C_Str(), importDir));

			if(aMat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &texPath) == aiReturn_SUCCESS)
				mat->Set(MaterialPropertyIds::Roughness, TextureFile::Import(scene, texPath.C_Str(), importDir));

			return mat;
		}



		void ImportMesh(std::shared_ptr<Model> model, aiMesh* aMesh)
		{
			// positions
			std::vector<float3> positions(aMesh->mNumVertices, make_float3(0));
			if(aMesh->mVertices)
				memcpy(positions.data(), aMesh->mVertices, aMesh->mNumVertices * sizeof(float3));

			// normals
			std::vector<float3> normals(aMesh->mNumVertices, make_float3(0));
			if(aMesh->mNormals)
				memcpy(normals.data(), aMesh->mNormals, aMesh->mNumVertices * sizeof(float3));

			// texcoords
			std::vector<float2> texcoords(aMesh->mNumVertices, make_float2(0));
			if(aMesh->mTextureCoords[0])
			{
				aiVector3D* src = aMesh->mTextureCoords[0];
				for(uint32_t i = 0; i < aMesh->mNumVertices; i++)
				{
					texcoords[i] = make_float2(src->x, src->y);
					src++;
				}
			}

			// indices
			std::vector<uint3> indices;
			indices.reserve(aMesh->mNumFaces);
			for(unsigned int i = 0; i < aMesh->mNumFaces; i++)
			{
				aiFace f = aMesh->mFaces[i];
				if(f.mNumIndices != 3)
					Logger::Error("Encountered non-triangulated face during import");
				else
					indices.push_back(make_uint3(f.mIndices[0], f.mIndices[1], f.mIndices[2]));
			}

			// add the mesh
			model->AddMesh(positions, normals, texcoords, indices, aMesh->mMaterialIndex);
		}
	}



	const std::vector<FileInfo>& ModelFile::SupportedFormats()
	{
		static std::vector<FileInfo> result;
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

				FileInfo fi;
				fi.name = ext;
				fi.description = ext;
				fi.ext = ext;
				result.push_back(fi);
			}
		}
		return result;
	}



	bool ModelFile::Supports(const std::string filePath)
	{
		static std::vector<FileInfo> formats = SupportedFormats();
		const std::string ext = ToLower(FileExtension(filePath).substr(1)); // lowercase extension without dot
		return std::find_if(formats.begin(), formats.end(), [ext](FileInfo fi) { return fi.ext == ext; }) != formats.end();
	}



	std::shared_ptr<Tracer::Model> ModelFile::Import(Scene* scene, const std::string& filePath, const std::string& name)
	{
		Stopwatch sw;
		const std::string globalPath = GlobalPath(filePath);

#if MODEL_CACHE_ENABLED
		// check the cache
		if(std::shared_ptr<Model> model = LoadFromCache(scene, globalPath, name))
		{
			Logger::Info("Loaded \"%s\" from cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());
			return model;
		}
#endif

		const std::string importDir = Directory(globalPath);

		// attach log stream
		static Assimp::Logger* defaultLogger = nullptr;
		if(!defaultLogger)
		{
			defaultLogger = Assimp::DefaultLogger::create("assimp.log", Assimp::Logger::NORMAL);
			defaultLogger->attachStream(new ImportLogStream<Logger::Severity::Debug>(), Assimp::Logger::Debugging);
			defaultLogger->attachStream(new ImportLogStream<Logger::Severity::Info>(), Assimp::Logger::Info);
			defaultLogger->attachStream(new ImportLogStream<Logger::Severity::Warning>(), Assimp::Logger::Warn);
			defaultLogger->attachStream(new ImportLogStream<Logger::Severity::Error>(), Assimp::Logger::Err);
		}

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
		{
			Logger::Error("Invalid Assimp import flags.");
			return nullptr;
		}

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
		std::shared_ptr<Model> model = std::make_shared<Model>(globalPath, name);

		// import materials
		std::map<std::string, std::shared_ptr<Texture>> textures;
		for(uint32_t i = 0; i < aScene->mNumMaterials; i++)
			model->AddMaterial(ImportMaterial(scene, importDir, aScene->mMaterials[i]));

		// import meshes
		uint32_t polyCount = 0;
		for(uint32_t i = 0; i < aScene->mNumMeshes; i++)
		{
			ImportMesh(model, aScene->mMeshes[i]);
			polyCount += aScene->mMeshes[i]->mNumFaces;
		}

		Logger::Info("Imported \"%s\" in %s", globalPath.c_str(), sw.ElapsedString().c_str());
		Logger::Debug("  Meshes   : %s", ThousandSeparators(aScene->mNumMeshes).c_str());
		Logger::Debug("  Materials: %s", ThousandSeparators(model->Materials().size()).c_str());
		Logger::Debug("  Textures : %s", ThousandSeparators(textures.size()).c_str());
		Logger::Debug("  Polygons : %s", ThousandSeparators(polyCount).c_str());

#if MODEL_CACHE_ENABLED
		sw.Reset();
		SaveToCache(model, globalPath);
		Logger::Debug("Saved \"%s\" to cache in %s", globalPath.c_str(), sw.ElapsedString().c_str());
#endif

		return model;
	}
}
