#include "MatVecDeclaration.h"
#include "Common Macros.h"

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cerrno>
#include <cstring>

MatVec::MatVec(double value, bool Host)
{
	this->Host = true;
	this->ElementCount = 0;
	this->Dimension = { 1, 1 };
	if (Host == true)
	{
		this->Data = (double*) std::malloc(sizeof(double));
		if (this->Data == NULL)
		{
			throw std::runtime_error(strerror(errno));
		}
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, sizeof(double));
		CHECK_CUDA_ERROR(t)
	}
	this->Data[0] = value;
}

MatVec::MatVec(std::vector<double>& values, bool Host)
{
	if (values.size() <= 0)
	{
		throw std::runtime_error("Vector of values has a size of 0 or less");
	}
	this->Host = Host;
	this->ElementCount = values.size();
	this->Dimension = { this->ElementCount, 1 };
	if (Host == true)
	{
		this->Data = (double*)std::malloc(sizeof(double) * this->ElementCount);
		if (this->Data == NULL)
		{
			throw std::runtime_error(strerror(errno));
		}
		std::memcpy(this->Data, values.data(), sizeof(double) * this->ElementCount);
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, sizeof(double));
		CHECK_CUDA_ERROR(t)
		t = cudaMemcpy(this->Data, values.data(), sizeof(double) * this->ElementCount, cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR(t)
	}
}

MatVec::MatVec(std::vector<std::vector<double>>& values, bool Host)
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
	this->ElementCount = rowCount * colCount;
	this->Dimension = { rowCount, colCount };
	this->Host = Host;
	if (Host == true)
	{
		this->Data = (double*)malloc(rowCount * colCount * sizeof(double));

		double* temp = this->Data;
		size_t step = rowCount * sizeof(double);
		for (std::vector<double> inner : values)
		{
			std::memcpy(temp, inner.data(), step);
			temp = temp + step;
		}
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, rowCount * colCount * sizeof(double));
		CHECK_CUDA_ERROR(t)
		double* temp = this->Data;
		size_t step = rowCount * sizeof(double);
		for (std::vector<double> inner : values)
		{
			t = cudaMemcpy(temp, inner.data(), step, cudaMemcpyHostToDevice);
			temp = temp + step;
			CHECK_CUDA_ERROR(t)
		}
	}
}

MatVec::MatVec(unsigned long long int row, unsigned long long int col, bool Host)
{
	this->Host = Host;
	this->ElementCount = row * col;
	this->Dimension = { row, col };
	if (Host == true)
	{
		this->Data = (double*)malloc(this->ElementCount * sizeof(double));
		CHECK_MALLOC_REQUEST(this->Data)
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, this->ElementCount * sizeof(double));
		CHECK_CUDA_ERROR(t)
	}
}

MatVec::~MatVec()
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