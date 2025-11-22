using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Lazulite.Kernels;

public static class ElementwiseKernels
{
    #region Binary
    public static void ElementwiseAddKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] + b[index];
    
    public static void ElementwiseSubtractKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] - b[index];
    
    public static void ElementwiseMultiplyKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] * b[index];
    
    public static void ElementwiseDivideKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] / b[index];
    
    public static void ElementwiseModuloKernel(Index1D index, ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] % b[index];
    
    public static void ElementwisePowerKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = XMath.Pow(a[index], b[index]);
    #endregion
    
    #region Unary
    public static void ElementwiseExpKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = XMath.Exp(a[index]);
    
    public static void ElementwiseLogKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = XMath.Log(a[index]);
    
    public static void ElementwiseSqrtKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = XMath.Sqrt(a[index]);
    
    public static void ElementwiseAbsKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = XMath.Abs(a[index]);
    
    public static void ElementwiseNegateKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = -a[index];
    #endregion

    #region Weird Ones
    public static void ElementwiseScalarPowerKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = XMath.Pow(a[index], b[0]);

    public static void ElementwiseScalarMultiplyKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a,
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] * b[0];
    
    public static void ElementwiseScalarDivideKernel(
        Index1D index, ArrayView1D<double, Stride1D.Dense> a,
        ArrayView1D<double, Stride1D.Dense> b, ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] / b[0];
    #endregion
}