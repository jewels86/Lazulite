using System.Linq.Expressions;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Jewels.Lazulite.Kernels;

public static class ElementwiseKernels
{
    #region Binary
    public static void ElementwiseAddKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = a[index] + b[index];
    
    public static void ElementwiseSubtractKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = a[index] - b[index];
    
    public static void ElementwiseMultiplyKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = a[index] * b[index];
    
    public static void ElementwiseDivideKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = a[index] / b[index];
    
    public static void ElementwiseModuloKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = a[index] % b[index];
    
    public static void ElementwisePowerKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Pow(a[index], b[index]);
    
    public static void ElementwiseMaxKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Max(a[index], b[index]);
    #endregion
    
    #region Unary
    public static void ElementwiseExpKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Exp(a[index]);
    
    public static void ElementwiseLogKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Log(a[index]);
    
    public static void ElementwiseSqrtKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Sqrt(a[index]);
    
    public static void ElementwiseAbsKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Abs(a[index]);
    
    public static void ElementwiseNegateKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = -a[index];
    
    public static void ElementwiseTanhKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Tanh(a[index]);
    
    public static void ElementwiseSech2Kernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = 1 / XMath.Cosh(a[index]);
    
    public static void ElementwiseNaturalLogKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Log(a[index], XMath.E);
    #endregion

    #region Weird Ones
    public static void ElementwiseScalarPowerKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Pow(a[index], b[0]);

    public static void ElementwiseScalarMultiplyKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = a[index] * b[0];
    
    public static void ElementwiseScalarDivideKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = a[index] / b[0];
    
    public static void ElementwiseScalarMaxKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> result) =>
        result[index] = XMath.Max(a[index], b[0]);
    
    public static void ElementwiseFloatPowerKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> result, float power) =>
        result[index] = XMath.Pow(a[index], power);
    
    public static void ElementwiseFloatMultiplyKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> result, float b) =>
        result[index] = a[index] * b;

    public static void ElementwiseFloatMaxKernel(
        Index1D index, ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> result, float b) =>
        result[index] = XMath.Max(a[index], b);
    #endregion
}