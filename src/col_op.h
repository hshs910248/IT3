namespace inplace {

namespace _3d {

template<typename F, typename T>
void col_op(F fn, T* data, int d1, int d2, int d3);

}

namespace _2d {

template<typename F, typename T>
void col_op(F fn, T* data, int d1, int d2);

}

}
