using System.Linq.Expressions;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Lazulite;

public static partial class Operations
{
    private static readonly Dictionary<int, CuBlas?> _cublasHandles = [];

    public static void Initialize()
    {
        foreach (var accelerator in Compute.Accelerators) GetCuBlas(accelerator.AcceleratorIndex());
        ApxyKernels = Compute.Load((
            Index1D i,
            ArrayView1D<float, Stride1D.Dense> x,
            ArrayView1D<float, Stride1D.Dense> y,
            float alpha) => y[i] = (alpha * x[i]) + y[i]);
    }

    public static void Cleanup()
    {
        foreach (var handle in _cublasHandles) handle.Value?.Dispose();
        _cublasHandles.Clear();
    }

    public static CuBlas? GetCuBlas(int aidx)
    {
        if (_cublasHandles.TryGetValue(aidx, out var blas) || Compute.Accelerators[aidx] is not CudaAccelerator cudaAccelerator) return blas;
        try
        {
            blas = new CuBlas(cudaAccelerator);
            _cublasHandles[aidx] = blas;
        }
        catch (Exception ex) { _cublasHandles[aidx] = null; }
        return blas;
    }
}