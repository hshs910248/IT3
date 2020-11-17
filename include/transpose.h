#pragma once

#include <cstdint>
#include <cstddef>

namespace inplace {

	void transpose(void* data, int rank, void* dim, int* permutation, size_t sizeofType);

}
