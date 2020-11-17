namespace inplace {

namespace _2d {

template<typename F, typename T>
void row_gather_op(F fn, T* data, int d1, int d2);

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int d1, int d2);

}

}
