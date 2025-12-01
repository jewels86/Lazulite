using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite.Kernels;

public static class MatrixKernels
{
    public static void MatrixMultiplyKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> result,
        int m, int k, int n,
        float alpha, float beta,
        int transposeA, int transposeB)
    {
        int row = index / n;
        int col = index % n;

        float sum = 0;
        for (int i = 0; i < k; i++)
        {
            int aIdx = transposeA == 1 ? (i * m + row) : (row * k + i);
            int bIdx = transposeB == 1 ? (col * k + i) : (i * n + col);
            sum += a[aIdx] * b[bIdx];
        }
    
        int resultIdx = row * n + col;
        result[resultIdx] = alpha * sum + beta * result[resultIdx];
    }

    public static void MatrixVectorMultiplyKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> matrix,
        ArrayView1D<float, Stride1D.Dense> vector,
        ArrayView1D<float, Stride1D.Dense> result,
        int m, int n,
        float alpha, float beta, int transposeMatrix) // matrix is m x n, vector is n (or m if transposed), result is m (or n if transposed)
    {
        if (transposeMatrix == 1)
        {
            // matrix^T is n x m, vector is m, result is n
            int col = index.X;
            if (col >= n) return;

            float sum = 0;
            for (int row = 0; row < m; row++) sum += matrix[row * n + col] * vector[row];
            result[col] = alpha * sum + beta * result[col];
        }
        else
        {
            int row = index.X;
            if (row >= m) return;

            float sum = 0;
            for (int col = 0; col < n; col++) sum += matrix[row * n + col] * vector[col];
            result[row] = alpha * sum + beta * result[row];
        }
    }

    public static void TransposeKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> matrix,
        ArrayView1D<float, Stride1D.Dense> result,
        int m, int n) // matrix is m x n, result is n x m 
    {
        if (index >= m * n) return;
    
        int row = index / n;
        int col = index % n;
    
        result[col * m + row] = matrix[row * n + col];
    }
}