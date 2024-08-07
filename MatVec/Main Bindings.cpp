#include <pybind11/pybind11.h>

#include "MatVecDeclaration.h"

namespace py = pybind11;

void MatVecClassBinding(py::module& m);

PYBIND11_MODULE(MatVec, m) 
{
	m.doc() = "A poor recreation of NumPy on CPU/GPU. (NVIDIA only so far)";
    MatVecClassBinding(m);
    

}