#include "MatVecDeclaration.h"

#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

std::string toStringHost(MatVec* mat)
{
    using namespace std;
    stringstream ss;
    ss << "Dimensions: { " << mat->Dimension[0] << ", " << mat->Dimension[1] << "}\n";
    ss << "Data memory location: " << mat->Data << "\n";
    ss << "[";
    for (size_t i = 0; i < mat->Dimension[0]; ++i) {
        ss << "[";
        for (size_t j = 0; j < mat->Dimension[1]; ++j)
        {
            ss << mat->Data[i * mat->Dimension[1] + j];
            if (j < mat->Dimension[1] - 1)
            {
                ss << ", ";
            }
        }
        ss << "]";
        if (i < mat->Dimension[0] - 1)
        {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

std::string toStringDevice(MatVec* mat)
{
    using namespace std;
    stringstream ss;
    ss << "Dimensions: { " << mat->Dimension[0] << ", " << mat->Dimension[1] << "}\n";
    ss << "Data memory location: " << mat->Data << "\n";
    ss << "[";
    for (size_t i = 0; i < mat->Dimension[0]; ++i) {
        ss << "[";
        for (size_t j = 0; j < mat->Dimension[1]; ++j)
        {
            ss << mat->Data[i * mat->Dimension[1] + j];
            if (j < mat->Dimension[1] - 1)
            {
                ss << ", ";
            }
        }
        ss << "]";
        if (i < mat->Dimension[0] - 1)
        {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}
std::string MatVec::toString()
{
    if (this->Host == true)
    {
        return toStringHost(this);
    }
    else
    {
        return toStringDevice(this);
    }
}