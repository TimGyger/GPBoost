/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifdef USE_CUDA_GP

#include <GPBoost/GP_utils.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

namespace GPBoost {

    bool try_matmul_gpu(const den_mat_t& A, const den_mat_t& B, den_mat_t& C) {
        int M = A.rows(), K = A.cols(), N = B.cols();
        if (K != B.rows()) {
            Log::REInfo("[GPU] Dimension mismatch.");
            return false;
        }

        C.resize(M, N);

        const float* h_A = A.data();
        const float* h_B = B.data();
        float* h_C = C.data();

        float* d_A, * d_B, * d_C;
        cudaError_t cuda_stat;
        cublasStatus_t stat;
        cublasHandle_t handle;

        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);

        cuda_stat = cudaMalloc((void**)&d_A, size_A);
        if (cuda_stat != cudaSuccess) return false;
        cuda_stat = cudaMalloc((void**)&d_B, size_B);
        if (cuda_stat != cudaSuccess) return false;
        cuda_stat = cudaMalloc((void**)&d_C, size_C);
        if (cuda_stat != cudaSuccess) return false;

        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) return false;

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Note: cuBLAS is column-major, so we use B^T and A^T to compute C = A * B
        stat = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,               // Transposed dims due to column-major
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return false;
        }

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

        cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        Log::REInfo("[GPU] Matrix multiplication completed with cuBLAS.");
        return true;
    }

}  // namespace GPBoost

#endif  // USE_CUDA_GP
