using ILGPU;
using ILGPU.Runtime;
using static Jewels.Lazulite.Kernels.SimpleKernels;
using static Jewels.Lazulite.Kernels.ElementwiseKernels;
using static Jewels.Lazulite.Kernels.MatrixKernels;
using static Jewels.Lazulite.Kernels.VectorKernels;

namespace Jewels.Lazulite;

public partial class Compute
{
    #region Simple Kernels
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>[] FillKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>>[] ZeroKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] CopyKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] ConcatKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[] SliceKernels { get; private set; } = []; 
    #endregion
    #region Elementwise Kernels
    #region Binary
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseAddKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseSubtractKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseMultiplyKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseDivideKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseModuloKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwisePowerKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseMaxKernels { get; private set; } = [];
    #endregion
    #region Unary
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseExpKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseLogKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseSqrtKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseAbsKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseNegateKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseTanhKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseNaturalLogKernels { get; private set; } = [];
    #endregion
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseScalarPowerKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseScalarMultiplyKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseScalarDivideKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseScalarMaxKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ElementwiseFloatPowerKernels = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ElementwiseFloatMultiplyKernels = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ElementwiseFloatMaxKernels = [];
    #endregion
    #region Matrix Kernels
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int, int, float, float, int, int>[] MatrixMultiplyKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int, float, float, int>[] MatrixVectorMultiplyKernels { get; private set; } = [];
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[] TransposeKernels { get; private set; } = [];
    #endregion
    #region Vector Kernels
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int>[] OuterProductKernels { get; private set; } = [];
    #endregion

    private bool _initialized;
    private bool _coreInitialized;
    private bool _extraScalarInitialized;
    private bool _extraElementwiseInitialized;
    
    private Task? _initializeCoreTask;
    private Task? _initializeExtraScalarTask;
    private Task? _initializeExtraElementwiseTask;

    #region Initializers
    private void InitializeBootstrapKernels()
    {
        if (_initialized) return;
        FillKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>[Accelerators.Count];
        ZeroKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        CopyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        foreach (var kvp in Accelerators)
        {
            var (aidx, accelerator) = kvp;
            FillKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float>(FillKernel);
            ZeroKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(ZeroKernel);
            CopyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(CopyKernel);
        }
        _initialized = true;
    }
    public void InitializeCoreKernels(bool warmup = true)
    {
        if (_coreInitialized) return;
        ConcatKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        SliceKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[Accelerators.Count];

        ElementwiseAddKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseSubtractKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseDivideKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        ElementwiseSqrtKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseNegateKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        ElementwiseScalarPowerKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseScalarMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseScalarDivideKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        MatrixMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int, int, float, float, int, int>[Accelerators.Count];
        MatrixVectorMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int, float, float, int>[Accelerators.Count];
        TransposeKernels = new Action<Index1D,ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[Accelerators.Count];
        OuterProductKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int>[Accelerators.Count];

        foreach (var kvp in Accelerators)
        {
            var (aidx, accelerator) = kvp;
            ConcatKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ConcatKernel);
            SliceKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(SliceKernel);

            ElementwiseAddKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseAddKernel);
            ElementwiseSubtractKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseSubtractKernel);
            ElementwiseMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseMultiplyKernel);
            ElementwiseDivideKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseDivideKernel);

            ElementwiseSqrtKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseSqrtKernel);
            ElementwiseNegateKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseNegateKernel);

            ElementwiseScalarPowerKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarPowerKernel);
            ElementwiseScalarMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarMultiplyKernel);
            ElementwiseScalarDivideKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarDivideKernel);

            MatrixMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, float, float, int, int>(MatrixMultiplyKernel);
            MatrixVectorMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, float, float, int>(MatrixVectorMultiplyKernel);
            TransposeKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, int, int>(TransposeKernel);

            OuterProductKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>(OuterProductKernel);
        }
        
        if (warmup) WarmupCoreKernels();
        _coreInitialized = true;
    }

    public void InitializeExtraScalarKernels(bool warmup = true)
    {
        if (_extraScalarInitialized) return;
        
        ElementwiseScalarMaxKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseFloatPowerKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[Accelerators.Count];
        ElementwiseFloatMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[Accelerators.Count];
        ElementwiseFloatMaxKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[Accelerators.Count];

        foreach (var kvp in Accelerators)
        {
            var (aidx, accelerator) = kvp;
            ElementwiseScalarMaxKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseScalarMaxKernel);
            ElementwiseFloatPowerKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, float>(ElementwiseFloatPowerKernel);
            ElementwiseFloatMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, float>(ElementwiseFloatMultiplyKernel);
            ElementwiseFloatMaxKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, float>(ElementwiseFloatMaxKernel);
        }
        
        if (warmup) WarmupExtraScalarKernels();
        _extraScalarInitialized = true;
    }

    public void InitializeExtraElementwiseKernels(bool warmup = true)
    {
        if (_extraElementwiseInitialized) return;
        
        ElementwiseModuloKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwisePowerKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseMaxKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        ElementwiseExpKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseLogKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseAbsKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseTanhKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseNaturalLogKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        foreach (var kvp in Accelerators)
        {
            var (aidx, accelerator) = kvp;
            ElementwiseModuloKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseModuloKernel);
            ElementwisePowerKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwisePowerKernel);
            ElementwiseMaxKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseMaxKernel);

            ElementwiseExpKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseExpKernel);
            ElementwiseLogKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseLogKernel);
            ElementwiseAbsKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseAbsKernel);
            ElementwiseTanhKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseTanhKernel);
            ElementwiseNaturalLogKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ElementwiseNaturalLogKernel);
        }
        
        if (warmup) WarmupExtraElementwiseKernels();
        _extraElementwiseInitialized = true;
    }
    #endregion
    #region Warmups
    public void WarmupCoreKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);

            Call(ZeroKernels, dummy);
            Call(FillKernels, dummy, 1.0f);
            Call(CopyKernels, dummy, dummy);

            Call(ElementwiseAddKernels, dummy, dummy, dummy);
            Call(ElementwiseSubtractKernels, dummy, dummy, dummy);
            Call(ElementwiseMultiplyKernels, dummy, dummy, dummy);
            Call(ElementwiseDivideKernels, dummy, dummy, dummy);

            Call(ElementwiseSqrtKernels, dummy, dummy);
            Call(ElementwiseNegateKernels, dummy, dummy);

            Call(ElementwiseScalarPowerKernels, dummy, dummy, dummy);
            Call(ElementwiseScalarMultiplyKernels, dummy, dummy, dummy);
            Call(ElementwiseScalarDivideKernels, dummy, dummy, dummy);
            Synchronize(i);
        }
    }

    public void WarmupExtraScalarKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);

            Call(ElementwiseScalarMaxKernels, dummy, dummy, dummy);
            Call(ElementwiseFloatPowerKernels, dummy, dummy, 1.0f);
            Call(ElementwiseFloatMultiplyKernels, dummy, dummy, 1.0f);
            Call(ElementwiseFloatMaxKernels, dummy, dummy, 1.0f);
            Synchronize(i);
        }
    }

    public void WarmupExtraElementwiseKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);

            Call(ElementwiseModuloKernels, dummy, dummy, dummy);
            Call(ElementwisePowerKernels, dummy, dummy, dummy);
            Call(ElementwiseMaxKernels, dummy, dummy, dummy);

            Call(ElementwiseExpKernels, dummy, dummy);
            Call(ElementwiseLogKernels, dummy, dummy);
            Call(ElementwiseAbsKernels, dummy, dummy);
            Call(ElementwiseTanhKernels, dummy, dummy);
            Call(ElementwiseNaturalLogKernels, dummy, dummy);
            Synchronize(i);
        }
    }
    #endregion
    #region Async Initialization
    public void InitializeCoreKernelsAsync(bool warmup = true) => _initializeCoreTask = Task.Run(() =>
    {
        InitializeCoreKernels(warmup);
    });
    public void InitializeExtraScalarKernelsAsync(bool warmup = true) => _initializeExtraScalarTask = Task.Run(() => InitializeExtraScalarKernels(warmup));
    public void InitializeExtraElementwiseKernelsAsync(bool warmup = true) => _initializeExtraElementwiseTask = Task.Run(() => InitializeExtraElementwiseKernels(warmup));
    public void InitializeExtraKernelsAsync(bool warmup = true)
    {
        InitializeExtraScalarKernelsAsync(warmup);
        InitializeExtraElementwiseKernelsAsync(warmup);
    }
    public void InitializeKernelsAsync(bool warmup = true)
    {
        InitializeCoreKernelsAsync(warmup);
        InitializeExtraKernelsAsync(warmup);
    }
    
    public void WaitForInitializationAsync()
    {
        _initializeCoreTask?.Wait();
        _initializeExtraScalarTask?.Wait();
        _initializeExtraElementwiseTask?.Wait();
    }
    #endregion

    public void InitializeExtraKernels(bool warmup = true)
    {
        InitializeExtraScalarKernels(warmup);
        InitializeExtraElementwiseKernels(warmup);
    }
    public void InitializeKernels(bool warmup = true)
    {
        InitializeCoreKernels(warmup);
        InitializeExtraKernels(warmup);
    }

    public void ClearKernels()
    {
        _coreInitialized = false;
        _extraScalarInitialized = false;
        _extraElementwiseInitialized = false;
        
        ZeroKernels = [];
        FillKernels = [];
        CopyKernels = [];
        ConcatKernels = [];
        SliceKernels = [];
        
        ElementwiseAddKernels = [];
        ElementwiseSubtractKernels = [];
        ElementwiseMultiplyKernels = [];
        ElementwiseDivideKernels = [];
        ElementwiseSqrtKernels = [];
        ElementwiseNegateKernels = [];
        
        ElementwiseScalarPowerKernels = [];
        ElementwiseScalarMultiplyKernels = [];
        ElementwiseScalarDivideKernels = [];
        
        MatrixMultiplyKernels = [];
        MatrixVectorMultiplyKernels = [];
        OuterProductKernels = [];
        
        ElementwiseScalarMaxKernels = [];
        ElementwiseFloatPowerKernels = [];
        ElementwiseFloatMultiplyKernels = [];
        ElementwiseFloatMaxKernels = [];
        
        ElementwiseModuloKernels = [];
        ElementwisePowerKernels = [];
        ElementwiseMaxKernels = [];
        
        ElementwiseExpKernels = [];
        ElementwiseLogKernels = [];
        ElementwiseAbsKernels = [];
        ElementwiseTanhKernels = [];
        ElementwiseNaturalLogKernels = [];
    }
}