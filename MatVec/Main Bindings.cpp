#include <pybind11/pybind11.h>

#include "MatVecDeclaration.h"

namespace py = pybind11;

PYBIND11_MODULE(MatVec, m) 
{
	m.doc() = "A poor recreation of NumPy on CPU/GPU.";
    py::class_<MatVec>(m, "MatVec")
        .def(py::init<double, bool>(), py::arg("value"), py::arg("Host") = true)
        .def(py::init<std::vector<double>&, bool>(), py::arg("values"), py::arg("Host") = true)
        .def(py::init<std::vector<std::vector<double>>&, bool>(), py::arg("values"), py::arg("Host") = true)
        .def(py::init<unsigned long long int, unsigned long long int, bool>(), py::arg("row"), py::arg("col"), py::arg("Host") = true);

}