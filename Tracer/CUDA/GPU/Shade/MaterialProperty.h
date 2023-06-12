#pragma once

__global__ __launch_bounds__(128, 2)
void MaterialPropertyKernel(DECLARE_KERNEL_PARAMS)
{
	const int jobIx = threadIdx.x + (blockIdx.x * blockDim.x);
	if(jobIx >= pathCount)
		return;

	// gather path data
	const float4 O4 = pathStates[jobIx + (stride * 0)];

	// extract path data
	const uint32_t pathIx = PathIx(__float_as_uint(O4.w));
	const uint32_t pixelIx = pathIx % (resolution.x * resolution.y);

	// hit data
	const uint4 hd = hitData[pathIx];
	const float2 bary = DecodeBarycentrics(hd.x);
	const uint32_t instIx = hd.y;
	const uint32_t primIx = hd.z;

	// didn't hit anything
	if(primIx == ~0)
		return;

	// fetch intersection info
	Intersection intersection = {};
	HitMaterial hitMaterial = {};
	GetIntersectionAttributes(instIx, primIx, bary, intersection, hitMaterial);

	// cache propery value
	float3 c;
	switch(static_cast<MaterialPropertyIds>(renderFlags))
	{
	case MaterialPropertyIds::Anisotropic:
		c = make_float3(hitMaterial.anisotropic);
		break;

	case MaterialPropertyIds::Clearcoat:
		c = make_float3(hitMaterial.clearcoat);
		break;

	case MaterialPropertyIds::ClearcoatGloss:
		c = make_float3(hitMaterial.clearcoatGloss);
		break;

	case MaterialPropertyIds::Diffuse:
		c = hitMaterial.diffuse;
		break;

	case MaterialPropertyIds::Emissive:
		c = hitMaterial.emissive;
		break;

	case MaterialPropertyIds::Metallic:
		c = make_float3(hitMaterial.metallic);
		break;

	case MaterialPropertyIds::Normal:
		c = GetColor(MaterialData[intersection.matIx], MaterialPropertyIds::Normal, make_float2(intersection.texcoordX, intersection.texcoordY));
		break;

	case MaterialPropertyIds::Roughness:
		c = make_float3(hitMaterial.roughness);
		break;

	case MaterialPropertyIds::Sheen:
		c = make_float3(hitMaterial.sheen);
		break;

	case MaterialPropertyIds::SheenTint:
		c = make_float3(hitMaterial.sheenTint);
		break;

	case MaterialPropertyIds::Specular:
		c = make_float3(hitMaterial.specular);
		break;

	case MaterialPropertyIds::SpecularTint:
		c = make_float3(hitMaterial.specularTint);
		break;

	case MaterialPropertyIds::Subsurface:
		c = make_float3(hitMaterial.subsurface);
		break;

	default:
		c = make_float3(1, 0, 1);
		break;
	}

	// set result
	accumulator[pixelIx] += make_float4(c, 0);
}
