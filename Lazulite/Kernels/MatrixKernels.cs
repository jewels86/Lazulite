using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Kernels;

public static class MatrixKernels
{
    public static void MatrixMultiplyKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> result,
        int m, int k, int n) // a is m x k, b is k x n, result is m x n
    {
        if (index >= m * n) return;
    
        int row = index / n;
        int col = index % n;
    
        float sum = 0;
        for (int i = 0; i < k; i++)
        {
            sum += a[row * k + i] * b[i * n + col];
        }
    
        result[row * n + col] = sum;
    }
}