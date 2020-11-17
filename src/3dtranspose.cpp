#include <cstdio>
#include <cuda_runtime.h>
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "cudacheck.h"
#include "equations.h"
#include "row_op.h"

namespace inplace {

namespace _3d {
	void init_dims(void* dim, int& d1, int& d2, int& d3) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
		d3 = int_dim[2];
	}

	template<typename T>
	void transpose(T* data, void* dim, int type) {
		int d1, d2, d3;
		init_dims(dim, d1, d2, d3);
		switch (type) {
			case 213:
				return;
			case 132:
				return;
			case 312:
				_312::transpose(data, d1, d2, d3);
				return;
			case 231:
				_231::transpose(data, d1, d2, d3);
				return;
			case 321:
				_321::transpose(data, d1, d2, d3);
				return;
			default:
				printf("Invalid permutation\n");
				return;
		}
	}
	template void transpose(float*, void*, int);
	template void transpose(double*, void*, int);

	namespace _312 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3) {
			int dim[2];
			dim[0] = d1 * d2;
			dim[1] = d3;
			_2d::transpose(data, dim);
		}

	}

	namespace _231 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3) {
			int dim[2];
			dim[0] = d1;
			dim[1] = d2 * d3;
			_2d::transpose(data, dim);
		}

	}
	
	namespace _321 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3) {
			int dim[2];
			dim[0] = d1;
			dim[1] = d2 * d3;
			_2d::transpose(data, dim);
			_2d::row_gather_op(_321::row_shuffle(d2, d3), data, d2 * d3, d1);
			//_2d::row_scatter_op(_321::row_scatter_shuffle(d2, d3), data, d2 * d3, d1);
		}
	
	}
}
}