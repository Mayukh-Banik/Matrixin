#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

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

    MatVec(double value, bool Host = true)
    {
        this->Data = static_cast<double*>(malloc(sizeof(double)));
        *(this->Data) = value;
        this->ElementCount = 1;
        this->Dimension = { 1, 1 };
        this->Host = Host;
    }

    MatVec(std::vector<double>& values, bool Host = true)
    {
        if (values.size() == 0)
        {
            throw std::invalid_argument("Values needed within an array.");
        }
        this->Data = static_cast<double*>(malloc(values.size()));
        double* temp = this->Data;
        for (double val : values)
        {
            *temp++ = val;
        }
        this->ElementCount = values.size();
        this->Dimension = { values.size(), 1 };
        this->Host = Host;
    }

    MatVec(std::vector<std::vector<double>>& values, bool Host = true)
    {
        size_t rowCount = values.size();
        if (rowCount == 0) 
        {
            throw std::invalid_argument("Values needed within an array.");
        }
        size_t colCount = values[0].size();
        for (const auto& row : values) 
        {
            if (row.size() != colCount) 
            {
                throw std::invalid_argument("All inner vectors must have the same size.");
            }
        }
        this->Data = static_cast<double*>(malloc(rowCount * colCount * sizeof(double)));
        double* temp = this->Data;
        for (const auto& row : values) 
        {
            for (double val : row) 
            {
                *temp++ = val;
            }
        }
        this->ElementCount = rowCount * colCount;
        this->Dimension = { rowCount, colCount };
        this->Host = Host;
    }

    ~MatVec()
    {
        if (this->Host == true)
        {
            free(this->Data);
        }
        else
        {
            cudaFree(this->Data);
        }
    }
};

