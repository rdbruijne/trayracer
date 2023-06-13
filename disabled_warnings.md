# Disabled Warnings

4061, 4365, 4464, 4505, 4514, 4619, 4626, 4668, 4710, 4711, 4820, 5027, 5039, 5045, 5054, 6011

**CUDA Specific**
5247, 5248

## Warning Details

**4061**
enumerator 'identifier' in switch of enum 'enumeration' is not explicitly handled by a case label

**4365**
'action' : conversion from 'type_1' to 'type_2', signed/unsigned mismatch

**4464**
relative include path contains '..'

**4505**
'function' : unreferenced local function has been removed

**4514**
'function' : unreferenced inline function has been removed

**4626**
'derived class' : assignment operator was implicitly defined as deleted because a base class assignment operator is inaccessible or deleted

**4619**
#pragma warning : there is no warning number 'number'

**4668**
'symbol' is not defined as a preprocessor macro, replacing with '0' for 'directives'

**4710**
'_function_' : function not inlined

**4711**
function 'function' selected for inline expansion

**4820**
'bytes' bytes padding added after construct 'member_name'

**5027**
'type': move assignment operator was implicitly defined as deleted

**5039**
'_function_': pointer or reference to potentially throwing function passed to `extern C` function under `-EHc`. Undefined behavior may occur if this function throws an exception.

**5045**
Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified

**5054**
operator 'operator-name': deprecated between enumerations of different types

**5247**
section 'section-name' is reserved for C++ dynamic initialization. Manually creating the section will interfere with C++ dynamic initialization and may lead to undefined behavior

**5248**
section 'section-name' is reserved for C++ dynamic initialization. Variables manually put into the section may be optimized out and their order relative to compiler generated dynamic initializers is unspecified

**5264**
'variable-name': 'const' variable is not used

**6011**
Dereferencing NULL pointer 'pointer-name'.
