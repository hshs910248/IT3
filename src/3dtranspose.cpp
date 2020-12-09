#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>
#include "2dtranspose.h"
#include "3dtranspose.h"
#include "util.h"
#include "equations.h"
#include "row_op.h"
#include "col_op.h"
#include "debug.h"
#include "gcd.h"

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
			case 312:
				_312::transpose(data, d1, d2, d3);
				return;
			case 231:
				_231::transpose(data, d1, d2, d3);
				return;
			case 213:
				_213::transpose(data, d1, d2, d3);
				return;
			case 321:
				_321::transpose(data, d1, d2, d3);
				return;
			case 132:
				_132::transpose(data, d1, d2, d3);
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
	
	namespace _213 {
		template<typename T>
		void c2r(T* data, int d1, int d2, int d3) {
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
				col_op(_2d::c2r::rotate(d2, b), data, d1, d2, d3);
			}
			row_gather_op(_2d::c2r::row_shuffle(d2, d1, c, k), data, d1, d2, d3);
			col_op(_2d::c2r::col_shuffle(d2, d1, c), data, d1, d2, d3);
		}
		
		template<typename T>
		void r2c(T* data, int d1, int d2, int d3) {
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
			
			col_op(_2d::r2c::col_shuffle(a, c, d2, q), data, d1, d2, d3);
			row_scatter_op(_2d::r2c::row_scatter_shuffle(d2, d1, c, k), data, d1, d2, d3);
			if (c > 1) {
				col_op(_2d::r2c::rotate(d2, b), data, d1, d2, d3);
			}
		}
		
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3) {
			size_t data_size = sizeof(T) * d1 * d2 * d3;
			prefetch(data, data_size);
			
			if (d1 * d2 <= d3) {
				_2d::row_gather_op(_213::row_shuffle(d1, d2), data, d1 * d2, d3);
			}
			else {
				if (d1 > d2) c2r(data, d1, d2, d3);
				else r2c(data, d2, d1, d3);
			}
		}
	
	}
	
	namespace _132 {
		template<typename T>
		void c2r(T* data, int d1, int d2, int d3) {
			PRINT("Doing C2R transpose\n");
			
			int c, t, k;
			extended_gcd(d2, d3, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d3/c, t, k);
			} else {
				k = t;
			}

			int a = d2 / c;
			int b = d3 / c;
			if (c > 1) {
				col_op(_2d::c2r::rotate(d2, b), data, d1, d2, d3);
			}
			row_gather_op(_2d::c2r::row_shuffle(d2, d3, c, k), data, d1, d2, d3);
			col_op(_2d::c2r::col_shuffle(d2, d3, c), data, d1, d2, d3);
		}
		
		template<typename T>
		void r2c(T* data, int d1, int d2, int d3) {
			PRINT("Doing R2C transpose\n");

			int c, t, q;
			extended_gcd(d3, d2, c, t);
			if (c > 1) {
				extended_gcd(d3/c, d2/c, t, q);
			} else {
				q = t;
			}
			
			int a = d2 / c;
			int b = d3 / c;
			
			int k;
			extended_gcd(d2, d3, c, t);
			if (c > 1) {
				extended_gcd(d2/c, d3/c, t, k);
			} else {
				k = t;
			}
			
			col_op(_2d::r2c::col_shuffle(a, c, d2, q), data, d1, d2, d3);
			row_scatter_op(_2d::r2c::row_scatter_shuffle(d2, d3, c, k), data, d1, d2, d3);
			if (c > 1) {
				col_op(_2d::r2c::rotate(d2, b), data, d1, d2, d3);
			}
		}
	
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3) {
			size_t data_size = sizeof(T) * d1 * d2 * d3;
			prefetch(data, data_size);
			
			if (d2 < d1 && d1 >= 64) { // d2d3 < d1d3
				_2d::col_op(_132::row_permute(d3, d2), data, d1, d2 * d3);
			}
			else if (d1 > 2) {
				if (d2 > d3) c2r(data, d1, d3, d2);
				else r2c(data, d1, d2, d3);
			}
			else {
				if (d2 >= d3) {
					_312::transpose(data, d1, d2, d3);
					_2d::row_gather_op(_213::row_shuffle(d3, d1), data, d3 * d1, d2);
				}
				else {
					_2d::row_gather_op(_213::row_shuffle(d1, d2), data, d1 * d2, d3);
					_231::transpose(data, d2, d1, d3);
				}
			}
		}
	}
	
	namespace _321 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3) {
			if (std::max(d1, d2 * d3) <= std::max(d1 * d2, d3)) {
				_231::transpose(data, d1, d2, d3);
				_2d::row_gather_op(_213::row_shuffle(d2, d3), data, d2 * d3, d1);
			}
			else {
				_213::transpose(data, d1, d2, d3);
				_312::transpose(data, d2, d1, d3);
			}
		}
	
	}
}
}