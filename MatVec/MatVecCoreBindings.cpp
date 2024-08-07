#include <pybind11/pybind11.h>
#include "MatVecDeclaration.h"

namespace py = pybind11;

void MatVecClassBinding(py::module& m)
{
    py::class_<MatVec>(m, "MatVec")
        .def(py::init<double, bool>(), py::arg("value"), py::arg("Host"),
            "Constructor for a scalar value.\n\nArgs:\n\tvalue (double): The sclar value\n\tHost (bool, default = True): Flag indicating whether to host on CPU memory or store it on GPU.")
        .def(py::init<std::vector<double>&, bool>(), py::arg("values"), py::arg("Host"),
            "Constructor for a 1-D vector.\n\nArgs:\n\tvalue (List[doubles]): Same as np.array()\n\tHost (bool, default = True): Flag indicating whether to host on CPU memory or store it on GPU.")
        .def(py::init<std::vector<std::vector<double>>&, bool>(), py::arg("values"), py::arg("Host"))
        .def(py::init<unsigned long long int, unsigned long long int, bool>(), py::arg("row"), py::arg("col"), py::arg("Host") = true)
        .def("__str__", &MatVec::toString)
        .def("__repr__", &MatVec::toString);
}