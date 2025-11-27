using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public partial class Compute
{
    public void MatrixMultiply(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> result,
        int m, int k, int n, bool noCuBlas = false,
        float alpha = 1.0f, float beta = 0.0f,
        bool transposeA = false, bool transposeB = false)
    {
        int aidx = a.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || result.Length < 1e6)
            Call(aidx, MatrixMultiplyKernels, a.IntExtent, a.View, b.View, result.View, m, k, n, alpha, beta, transposeA ? 1 : 0, transposeB ? 1 : 0);
        else
            blas.Gemm(
                transposeA ? CuBlasOperation.Transpose : CuBlasOperation.NonTranspose,
                transposeB ? CuBlasOperation.Transpose : CuBlasOperation.NonTranspose,
                m, n, k,
                alpha,
                a.View.BaseView, transposeA ? m : k,
                b.View.BaseView, transposeB ? k : n,
                beta,
                result.View.BaseView, n);
    }

    public void MatrixVectorMultiply(
        MemoryBuffer1D<float, Stride1D.Dense> matrix,
        MemoryBuffer1D<float, Stride1D.Dense> vector,
        MemoryBuffer1D<float, Stride1D.Dense> result,
        int m, int n, float alpha = 1.0f, float beta = 0.0f,
        bool transposeMatrix = false, bool noCuBlas = false) // matrix is m x n, vector is n, result is m
    {
        var aidx = matrix.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        int resultSize = transposeMatrix ? n : m;

        if (blas is null || noCuBlas || matrix.Length < 1e5)
            Call(aidx, MatrixVectorMultiplyKernels, resultSize, matrix.View, vector.View, result.View, m, n, alpha, beta, transposeMatrix ? 1 : 0);
        else
            blas.Gemv(
            transposeMatrix ? CuBlasOperation.Transpose : CuBlasOperation.NonTranspose,
            m, n, alpha,
            matrix.View.BaseView, n,
            vector.View.AsGeneral(), beta,
            result.View.AsGeneral());
    }
    
}