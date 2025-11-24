using System.Diagnostics.Contracts;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public static partial class Operations
{
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> ApxyKernels { get; private set; } = [];
    
    
    public static void Sum(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> result)
    {
        var aidx = a.AcceleratorIndex();
        Compute.Accelerators[aidx].Reduce<float, AddFloat>(
            Compute.GetStream(aidx),
            a.View, result.View);
    }

    public static void Dot(
        MemoryBuffer1D<float, Stride1D.Dense> a, 
        MemoryBuffer1D<float, Stride1D.Dense> b, 
        MemoryBuffer1D<float, Stride1D.Dense> result, bool noCuBlas = false)
    {
        var aidx = a.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || a.Length < 1e4)
        {
            var temp = Compute.GetLike(a);
            Compute.Call(aidx, Compute.ElementwiseMultiplyKernels, result.IntExtent, a.View, b.View, temp.View);
            Sum(temp, result);
            Compute.Return(temp);
        }
        else
            blas.Dot(a.View.AsGeneral(), b.View.AsGeneral(), result.View.BaseView);
    }

    public static void Axpy(
        float alpha,
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        bool noCuBlas = false)
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || x.Length < 1e5) Compute.Call(aidx, ApxyKernels, x.IntExtent, x.View, y.View, alpha);
        else blas.Axpy(alpha, x.View.AsGeneral(), y.View.AsGeneral());
    }

    public static void Scale(
        float alpha,
        MemoryBuffer1D<float, Stride1D.Dense> x,
        bool noCuBlas = false)
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || x.Length < 1e5)
            Compute.Call(aidx, Compute.ElementwiseFloatMultiplyKernels, x, x, alpha);
        else blas.Scal(alpha, x.View.AsGeneral());
    }

    public static void OuterProduct(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        MemoryBuffer1D<float, Stride1D.Dense> result,
        int m, int n, float alpha = 1.0f,
        bool noCuBlas = false) // x is m, y is n, result is m x n 
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || m * n < 1e4)
            Compute.Call(aidx, Compute.OuterProductKernels, result.IntExtent, x.View, y.View, result.View, m, n);
        else
            blas.Ger(
                m, n, alpha,
                x.View.AsGeneral(),
                y.View.AsGeneral(),
                result.View.BaseView, n);
    }
}