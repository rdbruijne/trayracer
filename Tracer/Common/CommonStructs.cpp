#include "CommonStructs.h"

std::string ToString(MaterialPropertyIds materialProperty)
{
	return std::string(magic_enum::enum_name(materialProperty));
}



std::string ToString(RenderModes renderMode)
{
	return std::string(magic_enum::enum_name(renderMode));
}
