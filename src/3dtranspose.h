#include <cstdint>

namespace inplace {
namespace _3d {
	template<typename T>
	void transpose(T* data, void* dim, int type);
	
	namespace _312 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3);
	}
	
	namespace _231 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3);
	}
	
	namespace _321 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3);
	}
	
	namespace _213 {
		template<typename T>
		void transpose(T* data, int d1, int d2, int d3);
	}
}
}