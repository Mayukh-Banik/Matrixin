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
	this->Dimension.push_back(1);
	this->Dimension.push_back(1);
	if (Host == true)
	{
		this->Data = (double*) std::malloc(DOUBLE_SIZE);
		if (this->Data == NULL)
		{
			throw std::runtime_error(strerror(errno));
		}
		this->Data[0] = value;
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, DOUBLE_SIZE);
		CHECK_CUDA_ERROR(t)
			cudaMemcpy(this->Data, &value, DOUBLE_SIZE, cudaMemcpyHostToDevice);
	}

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
		this->Data = (double*)std::malloc(DOUBLE_SIZE * this->ElementCount);
		if (this->Data == NULL)
		{
			throw std::runtime_error(strerror(errno));
		}
		std::memcpy(this->Data, values.data(), DOUBLE_SIZE * this->ElementCount);
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, DOUBLE_SIZE);
		CHECK_CUDA_ERROR(t)
		t = cudaMemcpy(this->Data, values.data(), DOUBLE_SIZE * this->ElementCount, cudaMemcpyHostToDevice);
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
		this->Data = (double*)malloc(rowCount * colCount * DOUBLE_SIZE);
		CHECK_MALLOC_REQUEST(this->Data)
		double* temp = this->Data;
		for (std::vector<double> inner : values)
		{
			std::memcpy(temp, inner.data(), rowCount * DOUBLE_SIZE);
			temp = temp + rowCount;
		}
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, rowCount * colCount * DOUBLE_SIZE);
		CHECK_CUDA_ERROR(t)
		double* temp = this->Data;
		size_t step = rowCount * DOUBLE_SIZE;
		for (std::vector<double> inner : values)
		{
			t = cudaMemcpy(temp, inner.data(), rowCount * DOUBLE_SIZE, cudaMemcpyHostToDevice);
			temp = temp + rowCount;
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
		this->Data = (double*)malloc(this->ElementCount * DOUBLE_SIZE);
		CHECK_MALLOC_REQUEST(this->Data)
	}
	else
	{
		cudaError t = cudaMalloc((void**)&this->Data, this->ElementCount * DOUBLE_SIZE);
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