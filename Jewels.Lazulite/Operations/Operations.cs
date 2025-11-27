using System.Linq.Expressions;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public  partial class Compute
{
    private readonly Dictionary<int, CuBlas?> _cublasHandles = [];
    
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ApxyKernels { get; private set; } = [];

    public void InitializeCuBlas()
    {
        foreach (var aidx in Accelerators.Keys) GetCuBlas(aidx);
        ApxyKernels = Load((
            Index1D i,
            ArrayView1D<float, Stride1D.Dense> x,
            ArrayView1D<float, Stride1D.Dense> y,
            float alpha) => y[i] = alpha * x[i] + y[i]);
    }

    public void CleanupCuBlas()
    {
        foreach (var handle in _cublasHandles) handle.Value?.Dispose();
        _cublasHandles.Clear();
    }

    public CuBlas? GetCuBlas(int aidx)
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