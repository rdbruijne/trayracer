#include "CommonStructs.h"

std::string ToString(RenderModes renderMode)
{
	return std::string(magic_enum::enum_name(renderMode));
}
