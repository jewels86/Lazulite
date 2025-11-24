using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public static partial class Operations
{
    public static void MatrixMultiply(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> result,
        int m, int k, int n, bool noCuBlas = false)
    {
        int aidx = a.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || result.Length < 1e6)
            Compute.Call(aidx, Compute.MatrixMultiplyKernels, result.IntExtent, a.View, b.View, result.View, m, k, n);
        else
            blas.Gemm(
                CuBlasOperation.NonTranspose,
                CuBlasOperation.NonTranspose,
                n, m, k,
                1.0f,
                b.View.BaseView, n,
                a.View.BaseView, k,
                0.0f,
                result.View.BaseView, n);
    }

    public static void MatrixVectorMultiply(
        MemoryBuffer1D<float, Stride1D.Dense> matrix,
        MemoryBuffer1D<float, Stride1D.Dense> vector,
        MemoryBuffer1D<float, Stride1D.Dense> result,
        int m, int n, float alpha = 1.0f, float beta = 0.0f, 
        bool noCuBlas = false) // matrix is m x n, vector is n, result is m
    {
        var aidx = matrix.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || matrix.Length < 1e5)
            Compute.Call(aidx, Compute.MatrixVectorMultiplyKernels, m, matrix.View, vector.View, result.View, m, n);
        else
            blas.Gemv(
            CuBlasOperation.NonTranspose,
            m, n, alpha,
            matrix.View.BaseView, n,
            vector.View.AsGeneral(), beta,
            result.View.AsGeneral());
    }
    
}