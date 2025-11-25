using System.Linq.Expressions;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public static partial class Compute
{
    private static readonly Dictionary<int, CuBlas?> _cublasHandles = [];

    public static void InitializeCuBlas()
    {
        foreach (var accelerator in Accelerators) GetCuBlas(accelerator.AcceleratorIndex());
        ApxyKernels = Load((
            Index1D i,
            ArrayView1D<float, Stride1D.Dense> x,
            ArrayView1D<float, Stride1D.Dense> y,
            float alpha) => y[i] = alpha * x[i] + y[i]);
    }

    public static void CleanupCuBlas()
    {
        foreach (var handle in _cublasHandles) handle.Value?.Dispose();
        _cublasHandles.Clear();
    }

    public static CuBlas? GetCuBlas(int aidx)
    {
        if (_cublasHandles.TryGetValue(aidx, out var blas) || Accelerators[aidx] is not CudaAccelerator cudaAccelerator) return blas;
        try
        {
            blas = new CuBlas(cudaAccelerator);
            _cublasHandles[aidx] = blas;
        }
        catch (Exception) { _cublasHandles[aidx] = null; }
        return blas;
    }
}