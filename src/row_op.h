namespace inplace {

namespace _3d {

namespace _132 {

template<typename F, typename T>
void row_gather_op(F fn, T* data, int d1, int d2, int d3);

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int d1, int d2, int d3);

}

namespace _213 {

template<typename F, typename T>
void row_gather_op(F fn, T* data, int d1, int d2, int d3);

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int d1, int d2, int d3);

}

}

namespace _2d {

template<typename F, typename T>
void row_gather_op(F fn, T* data, int d1, int d2);

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int d1, int d2);

}

}
