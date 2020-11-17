#include <cstdint>

namespace inplace {
namespace _2d {
	template<typename T>
	void transpose(T* data, void* dim);
	
	namespace c2r {
		template<typename T>
		void transpose(T* data, int d1, int d2);
	}

	namespace r2c {
		template<typename T>
		void transpose(T* data, int d1, int d2);
	}
}
}
