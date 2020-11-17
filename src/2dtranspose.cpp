#include <cuda_runtime.h>
#include "2dtranspose.h"
#include "debug.h"
#include "util.h"
#include "cudacheck.h"
#include "gcd.h"
#include "equations.h"
#include "col_op.h"
#include "row_op.h"

namespace inplace {

namespace _2d {
	void init_dims(void* dim, int& d1, int& d2) {
		int* int_dim = reinterpret_cast<int*>(dim);
		d1 = int_dim[0];
		d2 = int_dim[1];
	}

	template<typename T>
	void transpose(T* data, void* dim) {
		int d1, d2;
		init_dims(dim, d1, d2);
		size_t data_size = sizeof(T) * d1 * d2;
		prefetch(data, data_size);
		
		PRINT("(d1, d2) = (%d, %d)\n", d1, d2);
		if (d1 >= d2) {
			c2r::transpose(data, d1, d2);
		}
		else {
			r2c::transpose(data, d2, d1);
		}
	}
	template void transpose(float*, void*);
	template void transpose(double*, void*);

	namespace c2r {
		template<typename T>
		void transpose(T* data, int d1, int d2) {
			PRINT("Doing C2R transpose\n");
			
			int c, t, k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d1/c, t, k);
			} else {
				k = t;
			}

			int a = d2 / c;
			int b = d1 / c;
			if (c > 1) {
				col_op(c2r::rotate(d2, b), data, d1, d2);
			}
			row_gather_op(c2r::row_shuffle(d2, d1, c, k), data, d1, d2);
			col_op(c2r::col_shuffle(d2, d1, c), data, d1, d2);
		}
	}

	namespace r2c {
		template<typename T>
		void transpose(T* data, int d1, int d2) {
			PRINT("Doing R2C transpose\n");

			int c, t, q;
			extended_gcd(d1, d2, c, t);
			if (c > 1) {
				extended_gcd(d1/c, d2/c, t, q);
			} else {
				q = t;
			}
			
			int a = d2 / c;
			int b = d1 / c;
			
			int k;
			extended_gcd(d2, d1, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d1/c, t, k);
			} else {
				k = t;
			}
			
			col_op(r2c::col_shuffle(a, c, d2, q), data, d1, d2);
			row_scatter_op(r2c::row_scatter_shuffle(d2, d1, c, k), data, d1, d2);
			if (c > 1) {
				col_op(r2c::rotate(d2, b), data, d1, d2);
			}
		}
	}
}

}