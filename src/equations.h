#pragma once
#include "reduced_math.h"
#ifdef DEBUG
#include <string>
#endif

namespace inplace {

namespace _3d {

namespace _321 {
	struct row_shuffle {
		int d2;
		reduced_divisor d3;
		__host__
		row_shuffle(int _d2, int _d3) : d2(_d2), d3(_d3) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			unsigned int jdivd3, jmodd3;
			d3.divmod(j, jdivd3, jmodd3);
			return jdivd3 + jmodd3 * d2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};
	
	/*struct row_scatter_shuffle {
		int d3;
		int d2;
		__host__
		row_scatter_shuffle(int _d2, int _d3) : d2(_d2), d3(_d3) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			return j / d2 + (j % d2) * d3;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};*/
	
	struct row_scatter_shuffle {
		int d3;
		reduced_divisor d2;
		__host__
		row_scatter_shuffle(int _d2, int _d3) : d2(_d2), d3(_d3) {}
		
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		
		__host__ __device__
		int operator()(const int& j) {
			unsigned int jdivd2, jmodd2;
			d2.divmod(j, jdivd2, jmodd2);
			return jdivd2 + jmodd2 * d3;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};
}

}

namespace _2d {

namespace c2r {

	/*struct rotate {
		uint32_t m, b;
		
		__host__  rotate(uint32_t _m, uint32_t _b) : m(_m), b(_b) {}
		
		__host__ __device__
		uint32_t operator()(const uint32_t& i, const uint32_t& j) {
			return (i + j / b) % m;
		}

	#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Rotate");
			return name;
		}
	#endif
	};*/

	struct rotate {
		reduced_divisor m;
		reduced_divisor b;
		__host__  rotate(int _m, int _b) : m(_m), b(_b) {}
		__host__  rotate() : m(1), b(1) {}
		
		__host__ __device__
		int operator()(const int& i, const int& j) {
			return (int)m.mod(i + (int)b.div(j));
		}

		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Rotate");
			return name;
		}
		#endif
	};

	/*struct row_shuffle {
		uint32_t m, n, k, b, c;
		__host__
		row_shuffle(uint32_t _m, uint32_t _n, uint32_t _c, uint32_t _k) : m(_m), n(_n), k(_k),
												  b(_n/_c), c(_c) {}
												  
		uint32_t i;
		__host__ __device__ 
		void set_i(const uint32_t& _i) {
			i = _i;
		}
		__host__ __device__
		uint32_t f(const uint32_t& j) {
			uint32_t r = j + i * (n - 1);
			//The (int) casts here prevent unsigned promotion
			//and the subsequent underflow: c implicitly casts
			//int - unsigned int to
			//unsigned int - unsigned int
			//rather than to
			//int - int
			//Which leads to underflow if the result is negative.
			/*if (i - (int)c.mod(j) <= m - (int)c.get()) {
				return r;
			} else {
				return r + m;
			}
			uint64_t ipc = static_cast<uint64_t>(i) + static_cast<uint64_t>(c);
			uint64_t mpjmodc = static_cast<uint64_t>(m) + static_cast<uint64_t>(j % c);
			//if (i + c <= m + (j % c)) {
			if (ipc <= mpjmodc) {
				return r;
			} else {
				return r + m;
			}
		}
		
		__host__ __device__
		uint32_t operator()(const uint32_t& j) {
			uint32_t fij = f(j);
			uint32_t fijdivc, fijmodc;
			//c.divmod(fij, fijdivc, fijmodc);
			fijdivc = fij / c;
			fijmodc = fij % c;
			//The extra mod in here prevents overflowing 32-bit int
			/*uint32_t term_1 = b.mod(k * b.mod(fijdivc));
			uint32_t term_2 = ((int)fijmodc) * (int)b.get();
			uint64_t kfijdivc = static_cast<uint64_t>(k) * static_cast<uint64_t>(fijdivc % b);
			uint32_t term_1 = static_cast<uint32_t>(kfijdivc % b);
			uint32_t term_2 = fijmodc * b;
			return term_1 + term_2;
		}

	#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Row Shuffle");
			return name;
		}
	#endif
	};*/
	
	struct row_shuffle {
		int m, n, k;
		reduced_divisor_64 b;
		reduced_divisor c;
		__host__
		row_shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
													   b(_n/_c), c(_c) {}
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		__host__ __device__
		int f(const int& j) {
			int r = j + i * (n - 1);
			//The (int) casts here prevent unsigned promotion
			//and the subsequent underflow: c implicitly casts
			//int - unsigned int to
			//unsigned int - unsigned int
			//rather than to
			//int - int
			//Which leads to underflow if the result is negative.
			if (i - (int)c.mod(j) <= m - (int)c.get()) {
				return r;
			} else {
				return r + m;
			}
		}
		
		__host__ __device__
		int operator()(const int& j) {
			int fij = f(j);
			unsigned int fijdivc, fijmodc;
			c.divmod(fij, fijdivc, fijmodc);
			int term_1 = b.mod((long long)k * (long long)fijdivc);
			int term_2 = ((int)fijmodc) * (int)b.get();
			return term_1+term_2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};

	/*struct col_shuffle {
		uint32_t m, n, a;
		__host__ 
		col_shuffle(uint32_t _m, uint32_t _n, uint32_t _a) : m(_m), n(_n), a(_a) {}
		
		__host__ __device__
		uint32_t operator()(const uint32_t& i, const uint32_t& j) {
			//return (int)m.mod(i * n - (int)m.div(i * c) + j);
			return (j + i * n - (i / a)) % m;
		}
		
	#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Col Shuffle");
			return name;
		}
	#endif
	};*/
	
	struct col_shuffle {
		reduced_divisor m;
		int n, c;
		__host__ 
		col_shuffle(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
		__host__ __device__
		int operator()(const int& i, const int& j) {
			return (int)m.mod(i * n - (int)m.div(i * c) + j);
		}
		
		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Col Shuffle");
			return name;
		}
		#endif
	};

} // End of c2r

namespace r2c {

	/*struct col_shuffle {
		uint32_t m, a, c, q;
		
		__host__ 
		col_shuffle(uint32_t _m, uint32_t _a, uint32_t _c, uint32_t _q) : m(_m) , a(_a), c(_c), q(_q) {}
		
		__host__ __device__ __forceinline__
		uint32_t p(const uint32_t& i) {
			uint32_t cm1 = c - 1;
			uint32_t cm1pidivc = (cm1 + i) / c;
			uint64_t qcm1pidivc = static_cast<uint64_t>(q) * static_cast<uint64_t>(cm1pidivc);
			uint32_t term_1 = static_cast<uint32_t>(qcm1pidivc % a);
			
			uint64_t cm1i = static_cast<uint64_t>(cm1) * static_cast<uint64_t>(i);
			uint32_t cm1imodc = static_cast<uint32_t>(cm1i % c);
			uint32_t term_2 = cm1imodc * a;
			return term_1 + term_2;
			
		}
		
		__host__ __device__
		uint32_t operator()(const uint32_t& i, const uint32_t& j) {
			uint32_t idx = (i + m - (j % m)) % m;
			return p(idx);
		}
		
	#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Col Shuffle");
			return name;
		}
	#endif
	};*/
	
	
	struct col_shuffle {
		reduced_divisor a;
		reduced_divisor c;
		reduced_divisor m;
		int q;
		__host__ 
		col_shuffle(int _a, int _c, int _m, int _q) : a(_a) , c(_c), m(_m), q(_q) {}
		__host__ __device__ __forceinline__
		int p(const int& i) {
			int cm1 = (int)c.get() - 1;
			int term_1 = int(a.get()) * (int)c.mod(cm1 * i);
			int term_2 = int(a.mod(int(c.div(cm1+i))*q));
			return term_1 + term_2;
			
		}
		__host__ __device__
		int operator()(const int& i, const int& j) {
			int idx = m.mod(i + (int)m.get() - (int)m.mod(j));
			return p(idx);
		}
		
		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Col Shuffle");
			return name;
		}
		#endif
	};

	/*struct row_shuffle {
		uint32_t m, n, b;
		__host__
		row_shuffle(uint32_t _m, uint32_t _n, uint32_t _b) : m(_m), n(_n), b(_b) {}
		
		uint32_t i;
		__host__ __device__ 
		void set_i(const uint32_t& _i) {
			i = _i;
		}
		
		__host__ __device__
		uint32_t operator()(const uint32_t& j) {
			return ((i + j / b) % m + j * m) % n;
		}
		
	#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Row Shuffle");
			return name;
		}
	#endif
	};*/
	
	struct row_shuffle {
		reduced_divisor m;
		reduced_divisor n;
		reduced_divisor b;
		__host__
		row_shuffle(int _m, int _n, int _b) : m(_m), n(_n), b(_b) {}
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}    
		__host__ __device__
		int operator()(const int& j) {
			int r = m.mod(b.div(j) + i) + j * (int)m.get();
			return n.mod(r);
		}
		
		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Row Shuffle");
			return name;
		}
		#endif
	};
	
	struct row_scatter_shuffle {
		int m, n, k;
		reduced_divisor_64 b;
		reduced_divisor c;
		__host__
		row_scatter_shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
													   b(_n/_c), c(_c) {}
		int i;
		__host__ __device__ 
		void set_i(const int& _i) {
			i = _i;
		}
		__host__ __device__
		int f(const int& j) {
			int r = j + i * (n - 1);
			//The (int) casts here prevent unsigned promotion
			//and the subsequent underflow: c implicitly casts
			//int - unsigned int to
			//unsigned int - unsigned int
			//rather than to
			//int - int
			//Which leads to underflow if the result is negative.
			if (i - (int)c.mod(j) <= m - (int)c.get()) {
				return r;
			} else {
				return r + m;
			}
		}
		
		__host__ __device__
		int operator()(const int& j) {
			int fij = f(j);
			unsigned int fijdivc, fijmodc;
			c.divmod(fij, fijdivc, fijmodc);
			int term_1 = b.mod((long long)k * (long long)fijdivc);
			int term_2 = ((int)fijmodc) * (int)b.get();
			return term_1+term_2;
		}
		
		#ifdef DEBUG
			__host__
			std::string getName() {
				std::string name("Row Shuffle");
				return name;
			}
		#endif
	};

	/*struct rotate {
		uint32_t m, b;
		
		__host__  rotate(uint32_t _m, uint32_t _b) : m(_m), b(_b) {}
		
		__host__ __device__
		uint32_t operator()(const uint32_t& i, const uint32_t& j) {
			return (i + m - (j / b)) % m;
			//return (uint32_t)m.mod(i + (uint32_t)m.get() - (uint32_t)b.div(j));
		}
		
	#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Rotate");
			return name;
		}
	#endif
	};*/
	
	struct rotate {
		reduced_divisor m;
		reduced_divisor b;
		
		__host__  rotate(int _m, int _b) : m(_m), b(_b) {}
		
		__host__  rotate() : m(1), b(1) {}
		
		__host__ __device__
		int operator()(const int& i, const int& j) {
			return (int)m.mod(i + (int)m.get() - (int)b.div(j));
		}
		
		#ifdef DEBUG
		__host__
		std::string getName() {
			std::string name("Rotate");
			return name;
		}
	#endif
	};

}

/*namespace c2r {

typedef r2c::prepermuter scatter_postpermuter;

}

namespace r2c {

typedef c2r::postpermuter scatter_prepermuter;

}*/

}
}