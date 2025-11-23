using System.Linq.Expressions;
using System.Reflection.Emit;
using ILGPU.Algorithms;
using ILGPU.Runtime.Cuda;

namespace Lazulite;

public static partial class Operations
{
    private static readonly Dictionary<int, CuBlas> _cublasHandles = [];

    public static void Cleanup()
    {
        foreach (var handle in _cublasHandles) handle.Value.Dispose();
        _cublasHandles.Clear();
    }

    private static CuBlas GetCuBlas(int aidx)
    {
        if (!_cublasHandles.TryGetValue(aidx, out var blas))
        {
            if (Compute.Accelerators[aidx] is CudaAccelerator cudaAccelerator)
            {
                blas = new CuBlas(cudaAccelerator);
                _cublasHandles[aidx] = blas;
            }
            else throw new Exception("Only CUDA accelerators are supported for CuBLAS operations.");
        }
        return blas;
    }
}