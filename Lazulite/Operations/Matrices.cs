using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Lazulite;

public static partial class Operations
{
    public static void MatrixMultiply(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> result,
        int m, int k, int n) // a is m x k, b is k x n, result is m x n
    {
        var aidx = a.AcceleratorIndex();
        if (Compute.IsGpuAccelerator(aidx) && Compute.Accelerators[aidx] is CudaAccelerator cudaAccelerator)
        {
            var blas = GetCuBlas(aidx);
            blas.Gemm(
                CuBlasOperation.NonTranspose,
                CuBlasOperation.NonTranspose,
                n, m, k,
                1.0f, a.View, n, b.View.BaseView, k, 0.0f, result.View.BaseView, n);
        }
        else Compute.Call(aidx, Compute.MatrixMultiplyKernels, result.IntExtent, a.View, b.View, result.View, m, k, n);
    }
}