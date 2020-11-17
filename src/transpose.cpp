#include <cstdio>
#include "2dtranspose.h"
#include "3dtranspose.h"

namespace inplace {

void transpose(void* data, int rank, void* dim, int* permutation, size_t sizeofType) {
	if (rank > 3) {
		printf("Rank not supported\n");
		return;
	}
	if (rank == 2) {
		if (sizeofType == 4) _2d::transpose(reinterpret_cast<float*>(data), dim);
		else _2d::transpose(reinterpret_cast<double*>(data), dim);
		return;
	}
	int type = 0;
	for (int i = 0; i < rank; i++) {
		type = type * 10 + permutation[i] + 1;
	}
	if (rank == 3) {
		if (sizeofType == 4) _3d::transpose(reinterpret_cast<float*>(data), dim, type);
		else _3d::transpose(reinterpret_cast<double*>(data), dim, type);
		return;
	}
}

}