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
#include <cusparse.h>
//#include <cusolverDn.h>
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

        const double* h_A = A.data();
        const double* h_B = B.data();
        double* h_C = C.data();

        double* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
        cudaError_t cuda_stat;
        cublasStatus_t stat;
        cublasHandle_t handle;

        size_t size_A = M * K * sizeof(double);
        size_t size_B = K * N * sizeof(double);
        size_t size_C = M * N * sizeof(double);

        cuda_stat = cudaMalloc((void**)&d_A, size_A);
        if (cuda_stat != cudaSuccess) return false;
        cuda_stat = cudaMalloc((void**)&d_B, size_B);
        if (cuda_stat != cudaSuccess) {
            cudaFree(d_A);
            return false;
        }

        cuda_stat = cudaMalloc((void**)&d_C, size_C);
        if (cuda_stat != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_B);
            return false;
        }

        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return false;
        }

        const double alpha = 1.0;
        const double beta = 0.0;

        // cuBLAS performs: C = alpha * op(A) * op(B) + beta * C
        // We want: C = A * B
        // A: MxK, B: KxN, C: MxN
        // So op(A) = A, op(B) = B
        stat = cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
            M, N, K,                   // C is MxN, A is MxK, B is KxN
            &alpha,
            d_A, M,  // lda = leading dim of A = M (since column-major)
            d_B, K,  // ldb = leading dim of B = K
            &beta,
            d_C, M); // ldc = leading dim of C = M

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

    bool try_diag_times_dense_gpu(const vec_t& D, const den_mat_t& B, den_mat_t& C) {
        int M = B.rows();
        int N = B.cols();

        if (D.size() != M) {
            Log::REInfo("[GPU] Dimension mismatch between diagonal and matrix.");
            return false;
        }

        C.resize(M, N);

        // Host pointers
        const double* h_D = D.data();
        const double* h_B = B.data();
        double* h_C = C.data();

        // Device pointers
        double* d_D = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;

        cudaMalloc((void**)&d_D, M * sizeof(double));
        cudaMalloc((void**)&d_B, M * N * sizeof(double));
        cudaMalloc((void**)&d_C, M * N * sizeof(double));

        cudaMemcpy(d_D, h_D, M * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, M * N * sizeof(double), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Multiply: C = diag(D) * B (i.e., scale each row of B by D[i])
        // Use cuBLAS: d_C = diag(d_D) * d_B
        cublasStatus_t stat = cublasDdgmm(handle,
            CUBLAS_SIDE_LEFT, // Left = scale rows (use RIGHT to scale columns)
            M, N,
            d_B, M,
            d_D, 1, // stride = 1
            d_C, M);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            Log::REInfo("[GPU] cuBLAS Ddgmm failed.");
            cudaFree(d_D); cudaFree(d_B); cudaFree(d_C);
            cublasDestroy(handle);
            return false;
        }

        cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_D);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);

        Log::REInfo("[GPU] Diagonal x Dense matrix multiplication completed with cuBLAS.");
        return true;
    }

    bool try_sparse_dense_matmul_gpu(const sp_mat_rm_t& A, const den_mat_t& B, den_mat_t& C) {
        int M = A.rows(), K = A.cols(), N = B.cols();
        if (K != B.rows()) {
            Log::REInfo("[GPU] Dimension mismatch.");
            return false;
        }

        //C.resize(M, N);

        // Convert Eigen sparse matrix to CSR format (cuSPARSE prefers CSR)
        const int nnz = A.nonZeros();
        const int* h_csrOffsets = A.outerIndexPtr();  // Row pointers
        const int* h_columns = A.innerIndexPtr();     // Column indices
        const double* h_values = A.valuePtr();        // Non-zero values

        // Allocate device memory
        int* d_csrOffsets;
        int* d_columns;
        double* d_values;
        double* d_B;
        double* d_C;

        cudaMalloc((void**)&d_csrOffsets, (M + 1) * sizeof(int));
        cudaMalloc((void**)&d_columns, nnz * sizeof(int));
        cudaMalloc((void**)&d_values, nnz * sizeof(double));
        cudaMalloc((void**)&d_B, K * N * sizeof(double));
        cudaMalloc((void**)&d_C, M * N * sizeof(double));

        cudaMemcpy(d_csrOffsets, h_csrOffsets, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_columns, h_columns, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemset(d_C, 0, M * N * sizeof(double));

        // Create cuSPARSE handle and descriptors
        cusparseHandle_t handle;
        cusparseCreate(&handle);

        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;
        cusparseCreateCsr(&matA, M, K, nnz,
            d_csrOffsets, d_columns, d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateDnMat(&matB, K, N, K, d_B, CUDA_R_64F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&matC, M, N, M, d_C, CUDA_R_64F, CUSPARSE_ORDER_COL);

        const double alpha = 1.0;
        const double beta = 0.0;

        size_t bufferSize = 0;
        void* dBuffer = nullptr;
        cusparseSpMM_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);

        cusparseStatus_t stat = cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer);

        if (stat != CUSPARSE_STATUS_SUCCESS) {
            Log::REInfo("[GPU] cuSPARSE SpMM failed.");
            cusparseDestroySpMat(matA);
            cusparseDestroyDnMat(matB);
            cusparseDestroyDnMat(matC);
            cusparseDestroy(handle);
            cudaFree(dBuffer); cudaFree(d_csrOffsets); cudaFree(d_columns);
            cudaFree(d_values); cudaFree(d_B); cudaFree(d_C);
            return false;
        }

        // Copy result back to host
        cudaMemcpy(C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

        // Clean up
        cusparseDestroySpMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
        cusparseDestroy(handle);

        cudaFree(dBuffer);
        cudaFree(d_csrOffsets);
        cudaFree(d_columns);
        cudaFree(d_values);
        cudaFree(d_B);
        cudaFree(d_C);

        Log::REInfo("[GPU] Sparse x Dense matrix multiplication completed with cuSPARSE.");
        return true;
    }

    bool solve_lower_triangular_gpu(const chol_den_mat_t& chol, const den_mat_t& R_host, den_mat_t& X_host) {
        den_mat_t L_host = chol.matrixL();
        int n = L_host.rows();
        int m = R_host.cols();
        if (L_host.cols() != n || R_host.rows() != n) {
            return false;
        }

        X_host = R_host; // output matrix

        // Allocate device memory
        double* d_L = nullptr;
        double* d_X = nullptr;

        cudaMalloc(&d_L, n * n * sizeof(double));
        cudaMalloc(&d_X, n * m * sizeof(double));

        cudaMemcpy(d_L, L_host.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, X_host.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        const double alpha = 1.0;

        // Solve: L * X = R -> X = L^{-1} * R
        // L is lower-triangular, column-major
        // Left-side, lower-triangular, no transpose, non-unit diagonal
        cublasStatus_t stat = cublasDtrsm(
            handle,
            CUBLAS_SIDE_LEFT,      // Solve L * X = R
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,           // No transpose
            CUBLAS_DIAG_NON_UNIT,  // Assume general diagonal
            n,                     // number of rows of L and X
            m,                     // number of columns of X
            &alpha,                // Scalar alpha
            d_L, n,                // L, leading dimension n
            d_X, n                 // R becomes X, leading dimension n
        );

        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_L); cudaFree(d_X);
            cublasDestroy(handle);
            return false;
        }

        // Copy result back
        cudaMemcpy(X_host.data(), d_X, n * m * sizeof(double), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_L);
        cudaFree(d_X);
        cublasDestroy(handle);

        Log::REInfo("[GPU] Triangular solve with CUBLAS.");
        return true;
    }

    /*bool cholesky_cusolver_to_eigen(chol_den_mat_t& llt, const den_mat_t& A_input) {
        int N = A_input.rows();
        if (A_input.cols() != N) return false;

        // Step 1: Create cuSolver handle
        cusolverDnHandle_t handle;
        cusolverDnCreate(&handle);

        // Step 2: Allocate GPU memory
        double* d_A;
        cudaMalloc((void**)&d_A, sizeof(double) * N * N);
        cudaMemcpy(d_A, A_input.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice);

        // Step 3: Buffer size & workspace
        int work_size = 0;
        cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &work_size);
        double* work;
        cudaMalloc((void**)&work, sizeof(double) * work_size);
        int* dev_info;
        cudaMalloc((void**)&dev_info, sizeof(int));

        // Step 4: Perform Cholesky on GPU
        cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, dev_info);

        int dev_info_h = 0;
        cudaMemcpy(&dev_info_h, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (dev_info_h != 0) {
            cudaFree(d_A); cudaFree(work); cudaFree(dev_info);
            cusolverDnDestroy(handle);
            return false;
        }

        // Step 5: Copy Cholesky factor back
        Eigen::MatrixXd L(N, N);
        cudaMemcpy(L.data(), d_A, sizeof(double) * N * N, cudaMemcpyDeviceToHost);

        // Step 6: Store result into LLT object (only lower triangle is valid)
        llt.compute(L.selfadjointView<Eigen::Lower>());

        // Step 7: Cleanup
        cudaFree(d_A); cudaFree(work); cudaFree(dev_info);
        cusolverDnDestroy(handle);

        return true;
    }*/

}  // namespace GPBoost

#endif  // USE_CUDA_GP
