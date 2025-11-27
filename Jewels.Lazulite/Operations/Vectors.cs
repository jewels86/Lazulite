using System.Diagnostics.Contracts;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public  partial class Compute
{
    public  void Sum(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> result)
    {
        var aidx = a.AcceleratorIndex();
        Accelerators[aidx].Reduce<float, AddFloat>(
            GetStream(aidx),
            a.View, result.View);
    }
    

    public  void Dot(
        MemoryBuffer1D<float, Stride1D.Dense> a, 
        MemoryBuffer1D<float, Stride1D.Dense> b, 
        MemoryBuffer1D<float, Stride1D.Dense> result, bool noCuBlas = false)
    {
        var aidx = a.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || (a.Length < 1e4 && b.Length < 1e4))
        {
            var temp = GetLike(a);
            Call(ElementwiseMultiplyKernels, a, b, temp);
            Sum(temp, result);
            Return(temp);
        }
        else
            blas.Dot(a.View.AsGeneral(), b.View.AsGeneral(), result.View.BaseView);
    }

    public  MemoryBuffer1D<float, Stride1D.Dense> Dot(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b, bool noCuBlas = false) => Encase(a.AcceleratorIndex(), 1, r => Dot(a, b, r, noCuBlas));

    public  void Axpy(
        float alpha,
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        bool noCuBlas = false)
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || x.Length < 1e5) Call(ApxyKernels, x.View, y.View, alpha);
        else blas.Axpy(alpha, x.View.AsGeneral(), y.View.AsGeneral());
    }

    public  void Scale(
        float alpha,
        MemoryBuffer1D<float, Stride1D.Dense> x,
        bool noCuBlas = false)
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || x.Length < 1e5)
            Call(ElementwiseFloatMultiplyKernels, x, x, alpha);
        else blas.Scal(alpha, x.View.AsGeneral());
    }

    public  void OuterProduct(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        MemoryBuffer1D<float, Stride1D.Dense> result,
        int m, int n, float alpha = 1.0f,
        bool noCuBlas = false) // x is m, y is n, result is m x n 
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || m * n < 1e4)
            Call(aidx, OuterProductKernels, result.IntExtent, x.View, y.View, result.View, m, n);
        else
            blas.Ger(
                m, n, alpha,
                x.View.AsGeneral(),
                y.View.AsGeneral(),
                result.View.BaseView, n);
    }
    public  MemoryBuffer1D<float, Stride1D.Dense> OuterProduct(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        int m, int n, float alpha = 1.0f, bool noCuBlas = false) => 
        Encase(x.AcceleratorIndex(), m * n, r => OuterProduct(x, y, r, m, n, alpha, noCuBlas));

    public  MemoryBuffer1D<float, Stride1D.Dense> Concat(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Encase(a.AcceleratorIndex(), (int)(a.Length + b.Length), r => Call(ConcatKernels, a, b, r));
}