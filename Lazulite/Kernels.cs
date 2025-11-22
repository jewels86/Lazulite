using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using Lazulite.Kernels;
using static Lazulite.Kernels.SimpleKernels;
using static Lazulite.Kernels.ElementwiseKernels;

namespace Lazulite;

public static partial class Compute
{
    #region Simple Kernels
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, double>> FillKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>>> ZeroKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> CopyKernels { get; } = [];
    #endregion
    #region Elementwise Kernels
    #region Binary
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseAddKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseSubtractKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseMultiplyKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseDivideKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseModuloKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwisePowerKernels { get; } = [];
    #endregion
    #region Unary
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseExpKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseLogKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseSqrtKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseAbsKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseNegateKernels { get; } = [];
    #endregion
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseScalarPowerKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseScalarMultiplyKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseScalarDivideKernels = [];
    #endregion
    #region Helpers
    private static Task? _warmupTask;
    public static void WarmupKernelsAsync() => _warmupTask = Task.Run(WarmupKernels);
    public static void EnsureWarmup() => _warmupTask?.Wait();
    #endregion

    public static void InitializeKernels()
    {
        foreach (var accelerator in Accelerators)
        {
            #region Simple Kernels
            FillKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, double>(FillKernel));
            ZeroKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>>(ZeroKernel));
            CopyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(CopyKernel));
            #endregion
            #region Elementwise Kernels
            #region Binary
            ElementwiseAddKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseAddKernel));
            ElementwiseSubtractKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseSubtractKernel));
            ElementwiseMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseMultiplyKernel));
            ElementwiseDivideKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseDivideKernel));
            ElementwiseModuloKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseModuloKernel));
            ElementwisePowerKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwisePowerKernel));
            #endregion
            #region Unary
            ElementwiseExpKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseExpKernel));
            ElementwiseLogKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseLogKernel));
            ElementwiseSqrtKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseSqrtKernel));
            ElementwiseAbsKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseAbsKernel));
            ElementwiseNegateKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseNegateKernel));
            #endregion
            ElementwiseScalarPowerKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseScalarPowerKernel));
            ElementwiseScalarMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseScalarMultiplyKernel));
            ElementwiseScalarDivideKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseScalarDivideKernel));
            #endregion
        }
    }
    
    public static void WarmupKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);
            
            Call(i, ZeroKernels, dummy);
            Call(i, FillKernels, dummy, 1);
            Call(i, CopyKernels, dummy, dummy);
            Call(i, ElementwiseAddKernels, dummy, dummy, dummy);
            Call(i, ElementwiseSubtractKernels, dummy, dummy, dummy);
            Call(i, ElementwiseMultiplyKernels, dummy, dummy, dummy);
            Call(i, ElementwiseDivideKernels, dummy, dummy, dummy);
            Call(i, ElementwiseModuloKernels, dummy, dummy, dummy);
            Call(i, ElementwisePowerKernels, dummy, dummy, dummy);
            Call(i, ElementwiseExpKernels, dummy, dummy);
            Call(i, ElementwiseLogKernels, dummy, dummy);
            Call(i, ElementwiseSqrtKernels, dummy, dummy);
            Call(i, ElementwiseAbsKernels, dummy, dummy);
            Call(i, ElementwiseNegateKernels, dummy, dummy);
            Call(i, ElementwiseScalarPowerKernels, dummy, dummy, dummy);
            Call(i, ElementwiseScalarMultiplyKernels, dummy, dummy, dummy);
            Call(i, ElementwiseScalarDivideKernels, dummy, dummy, dummy);;
            Synchronize(i);
        }
    }
}