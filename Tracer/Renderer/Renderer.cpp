#include "Renderer/Renderer.h"

// Project
#include "CUDA/CudaDevice.h"
#include "GUI/MainGui.h"
#include "OpenGL/GLTexture.h"
#include "Optix/Denoiser.h"
#include "Optix/OptixError.h"
#include "Optix/OptixRenderer.h"
#include "Renderer/DeviceRenderer.h"
#include "Renderer/Scene.h"
#include "Resources/CameraNode.h"
#include "Resources/Instance.h"
#include "Resources/Material.h"
#include "Resources/Model.h"
#include "Resources/Texture.h"
#include "Renderer/Sky.h"
#include "Utility/LinearMath.h"
#include "Utility/Logger.h"
#include "Utility/Stopwatch.h"
#include "Utility/Utility.h"

// SPT
#include "CUDA/GPU/CudaFwd.h"

// Optix
#pragma warning(push)
#pragma warning(disable: 4061 4365 5039 6011 6387 26451)
#include "optix7/optix.h"
#include "optix7/optix_stubs.h"
#pragma warning(pop)

// CUDA
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// C++
#include <assert.h>
#include <set>
#include <string>
#include <thread>

namespace Tracer
{
	Renderer::Renderer()
	{
		// list devices
		int deviceCount = 0;
		CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
		Logger::Info("Found %i CUDA device%s:", deviceCount, (deviceCount > 1 ? "s" : ""));
		for(int i = 0; i < deviceCount; i++)
		{
			cudaDeviceProp devProps = {};
			CUDA_CHECK(cudaGetDeviceProperties(&devProps, i));
			Logger::Info(" - %s", devProps.name);
		}

		// init devices
		constexpr int deviceID = 0;
		mCudaDevice = std::make_shared<CudaDevice>(deviceID);

		// init device renderer
		mDeviceRenderer = std::make_shared<DeviceRenderer>(mCudaDevice);

		// create denoiser
		mDenoiser = std::make_shared<Denoiser>(mDeviceRenderer->Optix()->DeviceContext());

		// init kernel settings
		mKernelSettings;
		mKernelSettings.multiSample = 1;
		mKernelSettings.maxDepth = MaxTraceDepth;
		mKernelSettings.aoDist = 1.f;
		mKernelSettings.zDepthMax = 1.f;
		mKernelSettings.rayEpsilon = Epsilon;
	}



	Renderer::~Renderer()
	{
		// #TODO: proper cleanup
		if(mCudaGraphicsResource)
			CUDA_CHECK(cudaGraphicsUnregisterResource(mCudaGraphicsResource));
	}



	void Renderer::UpdateScene(Scene* scene)
	{
		if(mLastScene == scene && !scene->IsDirty())
		{
			// clear timers
			mRenderStats.buildTimeMs     = 0;

			mRenderStats.geoBuildTimeMs  = 0;
			mRenderStats.matBuildTimeMs  = 0;
			mRenderStats.skyBuildTimeMs  = 0;

			mRenderStats.geoUploadTimeMs = 0;
			mRenderStats.matUploadTimeMs = 0;
			mRenderStats.skyUploadTimeMs = 0;

			return;
		}

		// #NOTE: build order is important for dirty checks
		Stopwatch sw;

		// build geometry
		Stopwatch sw2;
		BuildGeometry(scene);
		mRenderStats.geoBuildTimeMs = sw2.ElapsedMs();

		// upload geometry
		sw2.Reset();
		UploadGeometry(scene);
		mRenderStats.geoUploadTimeMs = sw2.ElapsedMs();

		// build materials
		sw2.Reset();
		BuildMaterials(scene);
		mRenderStats.matBuildTimeMs = sw2.ElapsedMs();

		// upload materials
		sw2.Reset();
		UploadMaterials(scene);
		mRenderStats.matUploadTimeMs = sw2.ElapsedMs();

		// build sky
		sw2.Reset();
		BuildSky(scene);
		mRenderStats.skyBuildTimeMs = sw2.ElapsedMs();

		// upload sky
		sw2.Reset();
		UploadSky(scene);
		mRenderStats.skyUploadTimeMs = sw2.ElapsedMs();

		// reset accumulation
		Reset();

		// sync devices
		CUDA_CHECK(cudaDeviceSynchronize());

		mRenderStats.buildTimeMs = sw.ElapsedMs();

		// mark scene as clean
		scene->MarkClean();

		// set last scene
		mLastScene = scene;
	}



	void Renderer::RenderFrame(GLTexture* renderTexture)
	{
		const int2 texRes = renderTexture->Resolution();
		if(texRes.x == 0 || texRes.y == 0)
			return;

		// resize the buffer
		if(mDeviceRenderer->Resolution() != texRes)
			Resize(renderTexture);

		// determine render flags
		uint32_t renderFlags = 0;
		if(mRenderMode == RenderModes::MaterialProperty)
			renderFlags = static_cast<uint32_t>(mMaterialPropertyId);

		// render the frame
		mRenderTimeEvent.Start(mCudaDevice->Stream());
		mDeviceRenderer->RenderFrame(0, texRes.y, mKernelSettings, mRenderMode, renderFlags);
		mRenderTimeEvent.Stop(mCudaDevice->Stream());
		cudaStreamSynchronize(mCudaDevice->Stream());

		// denoise the frame
		DenoiseFrame();

		// upload the frame
		UploadFrame(renderTexture);

		// update timings
		mRenderStats.renderTimeMs = mRenderTimeEvent.Elapsed();
		mRenderStats.denoiseTimeMs = mDenoiseTimeEvent.Elapsed();

		// copy from devices
		mRenderStats.devices.clear();
		mRenderStats.devices.push_back(mDeviceRenderer->Statistics());
	}



	void Renderer::Reset()
	{
		mDeviceRenderer->Reset();
		mDenoiser->Reset();
	}



	bool Renderer::SaveRequested(std::string& path)
	{
		if(mSavePath.empty())
			return false;

		path = mSavePath;
		mSavePath.clear();
		return true;
	}



	void Renderer::SetCamera(CameraNode& camNode)
	{
		if(camNode.IsDirty())
		{
			mDeviceRenderer->SetCamera(camNode);
			Reset();
			camNode.MarkClean();
		}
	}



	void Renderer::SetRenderMode(RenderModes mode)
	{
		if (mRenderMode != mode)
		{
			mRenderMode = mode;
			Reset();
		}
	}



	void Renderer::SetMaterialPropertyId(MaterialPropertyIds id)
	{
		if (mMaterialPropertyId != id)
		{
			mMaterialPropertyId = id;
			if(mRenderMode == RenderModes::MaterialProperty)
				Reset();
		}
	}



	RayPickResult Renderer::PickRay(int2 pixelIndex)
	{
		return mDeviceRenderer->PickRay(pixelIndex);
	}



	void Renderer::Resize(GLTexture* renderTexture)
	{
		const int2 resolution = renderTexture->Resolution();

		// resize denoiser
		mDenoiser->Resize(resolution);

		// resize the renderer
		mDeviceRenderer->Resize(resolution);

		// update launch params
		Reset();

		// release the graphics resource
		if(mCudaGraphicsResource)
			CUDA_CHECK(cudaGraphicsUnregisterResource(mCudaGraphicsResource));
		CUDA_CHECK(cudaGraphicsGLRegisterImage(&mCudaGraphicsResource, renderTexture->ID(), GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
	}



	bool Renderer::ShouldDenoise() const
	{
#pragma warning(push)
#pragma warning(disable: 4061)
		switch(mRenderMode)
		{
		case RenderModes::AmbientOcclusion:
		case RenderModes::AmbientOcclusionShading:
		case RenderModes::DirectLight:
		case RenderModes::PathTracing:
			return mDenoiser->IsEnabled() && (mDeviceRenderer->SampleCount() >= mDenoiser->SampleTreshold());
			break;

		default:
			return false;
		}
#pragma warning(pop)
	}



	void Renderer::BuildGeometry(Scene* scene)
	{
		if(!scene)
			return;

		// flag to check for light rebuild
		bool rebuildLights = false;

		// build models
		const std::vector<std::shared_ptr<Model>>& models = scene->Models();
		std::vector<std::thread> modelBuilders;
		std::atomic_size_t modelIx = 0;
		for(size_t i = 0; i < std::thread::hardware_concurrency(); i++)
		{
			modelBuilders.push_back(std::thread([this, &modelIx, &models, &rebuildLights]()
			{
				size_t ix = modelIx++;
				while(ix < models.size())
				{
					std::shared_ptr<Model> m = models[ix];
					if(m->IsDirty())
					{
						if(m->IsDirty(false))
							m->Build();
						if(m->BuildLights())
							rebuildLights = true;
					}
					ix = modelIx++;
				}
			}));
		}
		for(std::thread& b : modelBuilders)
			b.join();

		// check instances
		for(const std::shared_ptr<Instance>& inst : scene->Instances())
		{
			rebuildLights = rebuildLights || (inst->IsDirty() && (inst->GetModel()->LightCount() > 0));
			inst->MarkClean();
		}

		// build lights
		if(rebuildLights)
			scene->GatherLights();
	}



	void Renderer::BuildMaterials(Scene* scene)
	{
		if(!scene || scene->MaterialCount() == 0)
		{
			mCudaMaterialOffsets.Free();
			mCudaMaterialData.Free();
			mCudaModelIndices.Free();

			SetCudaMatarialData(mCudaMaterialData.Ptr<CudaMatarial>());
			SetCudaMatarialOffsets(mCudaMaterialOffsets.Ptr<uint32_t>());
			SetCudaModelIndices(mCudaModelIndices.Ptr<uint32_t>());

			return;
		}

		uint32_t lastMaterialOffset = 0;
		std::vector<std::shared_ptr<Model>> parsedModels;

		mMaterialOffsets.clear();
		mMaterialOffsets.reserve(scene->Instances().size());

		mModelIndices.clear();
		mModelIndices.reserve(scene->Instances().size());

		parsedModels.reserve(scene->Instances().size());

		// gather materials to build
		mMaterials.clear();
		for(const std::shared_ptr<Instance>& inst : scene->Instances())
		{
			// find the model
			const std::shared_ptr<Model>& model = inst->GetModel();
			std::vector<std::shared_ptr<Model>>::iterator it = std::find(parsedModels.begin(), parsedModels.end(), model);
			if(it != parsedModels.end())
			{
				// model already parsed
				mModelIndices.push_back(static_cast<uint32_t>(std::distance(parsedModels.begin(), it)));
			}
			else
			{
				// add materials
				const std::vector<std::shared_ptr<Material>>& modelMats = model->Materials();
				mMaterials.insert(mMaterials.end(), modelMats.begin(), modelMats.end());

				// increment offsets
				mMaterialOffsets.push_back(lastMaterialOffset);
				lastMaterialOffset += static_cast<uint32_t>(model->Materials().size());

				mModelIndices.push_back(static_cast<uint32_t>(parsedModels.size()));
				parsedModels.push_back(model);
			}
		}

		// build the materials
		std::atomic_size_t matIx = 0;
		std::vector<std::thread> materialBuilders;
		for(size_t i = 0; i < std::thread::hardware_concurrency(); i++)
		{
			materialBuilders.push_back(std::thread([this, &matIx]()
			{
				size_t ix = matIx++;
				while(ix < mMaterials.size())
				{
					mMaterials[ix]->Build();
					ix = matIx++;
				}
			}));
		}
		for(std::thread& b : materialBuilders)
			b.join();
	}



	void Renderer::BuildSky(Scene* scene)
	{
		if(scene && scene->GetSky())
			scene->GetSky()->Build();
	}



	void Renderer::UploadGeometry(Scene* scene)
	{
		if(!scene || scene->InstanceCount() == 0)
		{
			// upload empty scene data
			mCudaMeshData.Free();
			mCudaInstanceInverseTransforms.Free();
			mCudaLightsBuffer.Free();

			SetCudaMeshData(nullptr);
			SetCudaInvTransforms(nullptr);
			SetCudaLights(nullptr);
			SetCudaLightCount(0);
			SetCudaLightEnergy(0);

			mDeviceRenderer->Optix()->BuildAccel({});
			return;
		}

		// upload models
		for(const std::shared_ptr<Model>& model : scene->Models())
			model->Upload(this);

		// place instances
		std::vector<OptixInstance> instances;
		std::vector<CudaMeshData> meshData;
		std::vector<float3x4> invTransforms;

		const size_t instanceCount = scene->InstanceCount();
		instances.reserve(instanceCount);
		meshData.reserve(instanceCount);
		invTransforms.reserve(instanceCount);

		uint32_t instanceId = 0;
		for(const std::shared_ptr<Instance>& inst : scene->Instances())
		{
			const std::shared_ptr<Model>& model = inst->GetModel();
			instances.push_back(model->InstanceData(instanceId++, inst->Transform()));
			meshData.push_back(model->CudaMesh());
			invTransforms.push_back(inverse(inst->Transform()));
		}

		// CUDA mesh data
		mCudaMeshData.UploadAsync(meshData, true);
		SetCudaMeshData(mCudaMeshData.Ptr<CudaMeshData>());

		// CUDA inverse instance transforms
		mCudaInstanceInverseTransforms.UploadAsync(invTransforms, true);
		SetCudaInvTransforms(mCudaInstanceInverseTransforms.Ptr<float4>());

		// upload lights
		const std::vector<LightTriangle>& lightData = scene->Lights();
		mCudaLightsBuffer.UploadAsync(lightData, true);
		SetCudaLights(mCudaLightsBuffer.Ptr<LightTriangle>());
		SetCudaLightCount(static_cast<int32_t>(lightData.size()));
		SetCudaLightEnergy(lightData.size() == 0 ? 0 : lightData.back().sumEnergy);

		// build Optix scene
		mDeviceRenderer->Optix()->BuildAccel(instances);
	}



	void Renderer::UploadMaterials(Scene* scene)
	{
		std::vector<CudaMatarial> materialData;
		materialData.reserve(mMaterials.size());
		for(const std::shared_ptr<Material>& material : mMaterials)
		{
			material->Upload(this);
			materialData.push_back(material->CudaMaterial());
		}

		// upload data
		mCudaMaterialOffsets.UploadAsync(mMaterialOffsets, true);
		mCudaMaterialData.UploadAsync(materialData, true);
		mCudaModelIndices.UploadAsync(mModelIndices, true);

		// assign to cuda
		SetCudaMatarialData(mCudaMaterialData.Ptr<CudaMatarial>());
		SetCudaMatarialOffsets(mCudaMaterialOffsets.Ptr<uint32_t>());
		SetCudaModelIndices(mCudaModelIndices.Ptr<uint32_t>());
	}



	void Renderer::UploadSky(Scene* scene)
	{
		if(!scene)
			return;

		std::shared_ptr<Sky> sky = scene->GetSky();
		if(sky && sky->IsOutOfSync())
		{
			sky->Upload(this);
			SetCudaSkyData(sky->CudaData().Ptr<SkyData>());
		}
	}



	void Renderer::DenoiseFrame()
	{
		if(!ShouldDenoise())
		{
			// empty timing so that we can call "Elapsed"
			mDenoiseTimeEvent.Start(mCudaDevice->Stream());
			mDenoiseTimeEvent.Stop(mCudaDevice->Stream());
			return;
		}

		if(mDeviceRenderer->SampleCount() >= (mDenoiser->SampleCount() * Phi))
		{
			mDenoiseTimeEvent.Start(mCudaDevice->Stream());
			mDenoiser->DenoiseFrame(mCudaDevice->Stream(),
									mDeviceRenderer->Resolution(),
									mDeviceRenderer->SampleCount(),
									mDeviceRenderer->ColorBuffer(),
									mDeviceRenderer->AlbedoBuffer(),
									mDeviceRenderer->NormalBuffer());
			mDenoiseTimeEvent.Stop(mCudaDevice->Stream());
		}
	}



	void Renderer::UploadFrame(GLTexture* renderTexture)
	{
		// copy to GL texture
		const int2 texRes = renderTexture->Resolution();
		cudaArray* cudaTexPtr = nullptr;
		const void* srcBuffer = ShouldDenoise() ? mDenoiser->DenoisedBuffer().Ptr() : mDeviceRenderer->ColorBuffer().Ptr();
		CUDA_CHECK(cudaGraphicsMapResources(1, &mCudaGraphicsResource, 0));
		CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaTexPtr, mCudaGraphicsResource, 0, 0));
		CUDA_CHECK(cudaMemcpy2DToArray(cudaTexPtr, 0, 0, srcBuffer, texRes.x * sizeof(float4), texRes.x * sizeof(float4), texRes.y, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &mCudaGraphicsResource, 0));
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
