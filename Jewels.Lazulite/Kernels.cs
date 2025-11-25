using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using Jewels.Lazulite.Kernels;
using static Jewels.Lazulite.Kernels.SimpleKernels;
using static Jewels.Lazulite.Kernels.ElementwiseKernels;
using static Jewels.Lazulite.Kernels.MatrixKernels;
using static Jewels.Lazulite.Kernels.VectorKernels;

namespace Jewels.Lazulite;

public static partial class Compute
{
    #region Simple Kernels
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>> FillKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>>> ZeroKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> CopyKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>> ConcatKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>> SliceKernels { get; } = []; 
    #endregion
    #region Elementwise Kernels
    #region Binary
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseAddKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseSubtractKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseMultiplyKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseDivideKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseModuloKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwisePowerKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseMaxKernels { get; } = [];
    #endregion
    #region Unary
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseExpKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseLogKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseSqrtKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseAbsKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseNegateKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseTanhKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseNaturalLogKernels { get; } = [];
    #endregion
    public readonly static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseScalarPowerKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseScalarMultiplyKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseScalarDivideKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseScalarMaxKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float>> ElementwiseFloatPowerKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float>> ElementwiseFloatMultiplyKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float>> ElementwiseFloatMaxKernels = [];
    #endregion
    #region Matrix Kernels
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int, int>> MatrixMultiplyKernels { get; } = [];
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int>> MatrixVectorMultiplyKernels { get; } = [];
    #endregion
    #region Vector Kernels
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int>> OuterProductKernels { get; } = [];
    #endregion

    private static bool _coreInitialized = false;
    private static bool _extraScalarInitialized = false;
    private static bool _extraElementwiseInitialized = false;

    public static void InitializeCoreKernels(bool warmup = true)
    {
        if (_coreInitialized) return;
        foreach (var accelerator in Accelerators)
        {
            FillKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float>(FillKernel));
            ZeroKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(ZeroKernel));
            CopyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(CopyKernel));
            ConcatKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>>(ConcatKernel));
            SliceKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>(SliceKernel));

            ElementwiseAddKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseAddKernel));
            ElementwiseSubtractKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseSubtractKernel));
            ElementwiseMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseMultiplyKernel));
            ElementwiseDivideKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseDivideKernel));
            
            ElementwiseSqrtKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseSqrtKernel));
            ElementwiseNegateKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseNegateKernel));
            
            ElementwiseScalarPowerKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarPowerKernel));
            ElementwiseScalarMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarMultiplyKernel));
            ElementwiseScalarDivideKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarDivideKernel));
            
            MatrixMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>(MatrixMultiplyKernel));
            MatrixVectorMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(MatrixVectorMultiplyKernel));
            
            OuterProductKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(OuterProductKernel));
        }
        
        if (warmup) WarmupCoreKernels();
        _coreInitialized = true;
    }

    public static void InitializeExtraScalarKernels(bool warmup = true)
    {
        if (_extraScalarInitialized) return;
        foreach (var accelerator in Accelerators)
        {
            ElementwiseScalarMaxKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarMaxKernel));
            ElementwiseFloatPowerKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, float>(ElementwiseFloatPowerKernel));
            ElementwiseFloatMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, float>(ElementwiseFloatMultiplyKernel));
            ElementwiseFloatMaxKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, float>(ElementwiseFloatMaxKernel));
        }
        
        if (warmup) WarmupExtraScalarKernels();
        _extraScalarInitialized = true;
    }

    public static void InitializeExtraElementwiseKernels(bool warmup = true)
    {
        if (_extraElementwiseInitialized) return;
        foreach (var accelerator in Accelerators)
        {
            ElementwiseModuloKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseModuloKernel));
            ElementwisePowerKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwisePowerKernel));
            ElementwiseMaxKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseMaxKernel));
            
            ElementwiseExpKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseExpKernel));
            ElementwiseLogKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseLogKernel));
            ElementwiseAbsKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseAbsKernel));
            ElementwiseTanhKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseTanhKernel));
            ElementwiseNaturalLogKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseNaturalLogKernel));
        }
        
        if (warmup) WarmupExtraElementwiseKernels();
        _extraElementwiseInitialized = true;
    }
    
    public static void WarmupCoreKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);
            
            Call(i, ZeroKernels, dummy);
            Call(i, FillKernels, dummy, 1.0f);
            Call(i, CopyKernels, dummy, dummy);
            
            Call(i, ElementwiseAddKernels, dummy, dummy, dummy);
            Call(i, ElementwiseSubtractKernels, dummy, dummy, dummy);
            Call(i, ElementwiseMultiplyKernels, dummy, dummy, dummy);
            Call(i, ElementwiseDivideKernels, dummy, dummy, dummy);
            
            Call(i, ElementwiseSqrtKernels, dummy, dummy);
            Call(i, ElementwiseNegateKernels, dummy, dummy);
            
            Call(i, ElementwiseScalarPowerKernels, dummy, dummy, dummy);
            Call(i, ElementwiseScalarMultiplyKernels, dummy, dummy, dummy);
            Call(i, ElementwiseScalarDivideKernels, dummy, dummy, dummy);
            Synchronize(i);
        }
    }

    public static void WarmupExtraScalarKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);
            
            Call(i, ElementwiseScalarMaxKernels, dummy, dummy, dummy);
            Call(i, ElementwiseFloatPowerKernels, dummy, dummy, 1.0f);
            Call(i, ElementwiseFloatMultiplyKernels, dummy, dummy, 1.0f);
            Call(i, ElementwiseFloatMaxKernels, dummy, dummy, 1.0f);
        }
    }

    public static void WarmupExtraElementwiseKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);
            
            Call(i, ElementwiseModuloKernels, dummy, dummy, dummy);
            Call(i, ElementwisePowerKernels, dummy, dummy, dummy);
            Call(i, ElementwiseMaxKernels, dummy, dummy, dummy);
            
            Call(i, ElementwiseExpKernels, dummy, dummy);
            Call(i, ElementwiseLogKernels, dummy, dummy);
            Call(i, ElementwiseAbsKernels, dummy, dummy);
            Call(i, ElementwiseTanhKernels, dummy, dummy);
            Call(i, ElementwiseNaturalLogKernels, dummy, dummy);
        }
    }

    public static void InitializeExtraKernels(bool warmup = true)
    {
        InitializeExtraScalarKernels(warmup);
        InitializeExtraElementwiseKernels(warmup);
    }
}