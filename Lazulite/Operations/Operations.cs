using System.Linq.Expressions;
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
    
    public static readonly Expression<Func<float, float, float>> AddOp = (a, b) => a + b;
    public static readonly Expression<Func<float, float, float>> SubtractOp = (a, b) => a - b;
    public static readonly Expression<Func<float, float, float>> MultiplyOp = (a, b) => a * b;
    public static readonly Expression<Func<float, float, float>> DivideOp = (a, b) => a / b;
    public static readonly Expression<Func<float, float, float>> ModuloOp = (a, b) => a % b;
    public static readonly Expression<Func<float, float, float>> PowerOp = (a, b) => XMath.Pow(a, b);
}