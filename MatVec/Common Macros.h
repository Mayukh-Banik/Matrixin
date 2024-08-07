#pragma once

#define CHECK_CUDA_ERROR(err) \
	if (err != cudaSuccess)	\
	{	\
		throw std::runtime_error(cudaGetErrorString(err));	\
	}

#define CHECK_MALLOC_REQUEST(x) \
	if (x == NULL)	\
	{	\
		throw std::runtime_error(strerror(errno));	\
	}

#define DOUBLE_SIZE sizeof(double)