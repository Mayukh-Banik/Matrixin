#pragma once

//#include <cstdlib>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <stdexcept>
#include <vector>

/**
 * @brief Main MatVec Class with constructors
 *
 */
class MatVec
{
public:

    double* Data = NULL;
    unsigned long long int ElementCount = 0;
    std::vector<unsigned long long int> Dimension = {};
    bool Host;

    MatVec(double value, bool Host = true);

    MatVec(std::vector<double>& values, bool Host = true);

    MatVec(std::vector<std::vector<double>>& values, bool Host = true);

    MatVec(unsigned long long int row, unsigned long long int col, bool Host = true);

    ~MatVec();
};

