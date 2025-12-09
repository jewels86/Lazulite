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
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>[] FillKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>>[] ZeroKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] CopyKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] ConcatKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[] SliceKernels { get; private set; } = []; 
    #endregion
    #region Elementwise Kernels
    #region Binary
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] AddKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] SubtractKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseMultiplyKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] DivideKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] MaxKernels { get; private set; } = [];
    #endregion
    #region Unary
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ExpKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] LogKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] SqrtKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] AbsKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] NegateKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] SinKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] CosKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] TanKernels { get; private set; } = [];
    #endregion
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ScalarPowerKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ScalarMultiplyKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ScalarDivideKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] ScalarMaxKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] FloatPowerKernels = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] FloatMultiplyKernels = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] FloatMaxKernels = [];
    #endregion
    #region Matrix Kernels
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int, float, float, int, int>[] MatrixMultiplyKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int, float, float, int>[] MatrixVectorMultiplyKernels { get; private set; } = [];
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[] TransposeKernels { get; private set; } = [];
    #endregion
    #region Vector Kernels
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int>[] OuterProductKernels { get; private set; } = [];
    #endregion

    private static bool _initialized;
    private static bool _coreInitialized;
    private static bool _extraScalarInitialized;
    private static bool _extraElementwiseInitialized;
    
    private static Task? _initializeCoreTask;
    private static Task? _initializeExtraScalarTask;
    private static Task? _initializeExtraElementwiseTask;

    #region Initializers
    private static void InitializeBootstrapKernels()
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
    public static void InitializeCoreKernels(bool warmup = true)
    {
        if (_coreInitialized) return;
        ConcatKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        SliceKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[Accelerators.Count];

        AddKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        SubtractKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ElementwiseMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        DivideKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        SqrtKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        NegateKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        ScalarPowerKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ScalarMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        ScalarDivideKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        MatrixMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int, int, int, float, float, int, int>[Accelerators.Count];
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

            AddKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(AddKernel);
            SubtractKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(SubtractKernel);
            ElementwiseMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ElementwiseMultiplyKernel);
            DivideKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(DivideKernel);

            SqrtKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(SqrtKernel);
            NegateKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(NegateKernel);

            ScalarPowerKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ScalarPowerKernel);
            ScalarMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ScalarMultiplyKernel);
            ScalarDivideKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ScalarDivideKernel);

            MatrixMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, float, float, int, int>(MatrixMultiplyKernel);
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

    public static void InitializeExtraScalarKernels(bool warmup = true)
    {
        if (_extraScalarInitialized) return;
        
        ScalarMaxKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        FloatPowerKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[Accelerators.Count];
        FloatMultiplyKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[Accelerators.Count];
        FloatMaxKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[Accelerators.Count];

        foreach (var kvp in Accelerators)
        {
            var (aidx, accelerator) = kvp;
            ScalarMaxKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ScalarMaxKernel);
            FloatPowerKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, float>(FloatPowerKernel);
            FloatMultiplyKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, float>(FloatMultiplyKernel);
            FloatMaxKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, float>(FloatMaxKernel);
        }
        
        if (warmup) WarmupExtraScalarKernels();
        _extraScalarInitialized = true;
    }

    public static void InitializeExtraElementwiseKernels(bool warmup = true)
    {
        if (_extraElementwiseInitialized) return;
        
        MaxKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];

        ExpKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        LogKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        AbsKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        SinKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        CosKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        TanKernels = new Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[Accelerators.Count];
        foreach (var kvp in Accelerators)
        {
            var (aidx, accelerator) = kvp;
            MaxKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(MaxKernel);

            ExpKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(ExpKernel);
            LogKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(LogKernel);
            AbsKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(AbsKernel);
            SinKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(SinKernel);
            CosKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(CosKernel);
            TanKernels[aidx] = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(TanKernel);
        }
        
        if (warmup) WarmupExtraElementwiseKernels();
        _extraElementwiseInitialized = true;
    }
    #endregion
    #region Warmups
    public static void WarmupCoreKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);

            Call(ZeroKernels, dummy);
            Call(FillKernels, dummy, 1.0f);
            Call(CopyKernels, dummy, dummy);

            Call(AddKernels, dummy, dummy, dummy);
            Call(SubtractKernels, dummy, dummy, dummy);
            Call(ElementwiseMultiplyKernels, dummy, dummy, dummy);
            Call(DivideKernels, dummy, dummy, dummy);

            Call(SqrtKernels, dummy, dummy);
            Call(NegateKernels, dummy, dummy);

            Call(ScalarPowerKernels, dummy, dummy, dummy);
            Call(ScalarMultiplyKernels, dummy, dummy, dummy);
            Call(ScalarDivideKernels, dummy, dummy, dummy);
            Synchronize(i);
        }
    }

    public static void WarmupExtraScalarKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);

            Call(ScalarMaxKernels, dummy, dummy, dummy);
            Call(FloatPowerKernels, dummy, dummy, 1.0f);
            Call(FloatMultiplyKernels, dummy, dummy, 1.0f);
            Call(FloatMaxKernels, dummy, dummy, 1.0f);
            Synchronize(i);
        }
    }

    public static void WarmupExtraElementwiseKernels()
    {
        for (int i = 0; i < Accelerators.Count; i++)
        {
            using var dummy = Get(i, 10);

            Call(MaxKernels, dummy, dummy, dummy);

            Call(ExpKernels, dummy, dummy);
            Call(LogKernels, dummy, dummy);
            Call(AbsKernels, dummy, dummy);
            Call(SinKernels, dummy, dummy);
            Call(CosKernels, dummy, dummy);
            Call(TanKernels, dummy, dummy);
            Synchronize(i);
        }
    }
    #endregion
    #region Async Initialization
    public static void InitializeCoreKernelsAsync(bool warmup = true) => _initializeCoreTask = Task.Run(() =>
    {
        InitializeCoreKernels(warmup);
    });
    public static void InitializeExtraScalarKernelsAsync(bool warmup = true) => _initializeExtraScalarTask = Task.Run(() => InitializeExtraScalarKernels(warmup));
    public static void InitializeExtraElementwiseKernelsAsync(bool warmup = true) => _initializeExtraElementwiseTask = Task.Run(() => InitializeExtraElementwiseKernels(warmup));
    public static void InitializeExtraKernelsAsync(bool warmup = true)
    {
        InitializeExtraScalarKernelsAsync(warmup);
        InitializeExtraElementwiseKernelsAsync(warmup);
    }
    public static void InitializeKernelsAsync(bool warmup = true)
    {
        InitializeCoreKernelsAsync(warmup);
        InitializeExtraKernelsAsync(warmup);
    }
    
    public static void WaitForInitializationAsync()
    {
        _initializeCoreTask?.Wait();
        _initializeExtraScalarTask?.Wait();
        _initializeExtraElementwiseTask?.Wait();
    }
    #endregion

    public static void InitializeExtraKernels(bool warmup = true)
    {
        InitializeExtraScalarKernels(warmup);
        InitializeExtraElementwiseKernels(warmup);
    }
    public static void InitializeKernels(bool warmup = true)
    {
        InitializeCoreKernels(warmup);
        InitializeExtraKernels(warmup);
    }

    public static void ClearKernels()
    {
        _coreInitialized = false;
        _extraScalarInitialized = false;
        _extraElementwiseInitialized = false;
        
        ZeroKernels = [];
        FillKernels = [];
        CopyKernels = [];
        ConcatKernels = [];
        SliceKernels = [];
        
        AddKernels = [];
        SubtractKernels = [];
        ElementwiseMultiplyKernels = [];
        DivideKernels = [];
        SqrtKernels = [];
        NegateKernels = [];
        
        ScalarPowerKernels = [];
        ScalarMultiplyKernels = [];
        ScalarDivideKernels = [];
        
        MatrixMultiplyKernels = [];
        MatrixVectorMultiplyKernels = [];
        OuterProductKernels = [];
        
        ScalarMaxKernels = [];
        FloatPowerKernels = [];
        FloatMultiplyKernels = [];
        FloatMaxKernels = [];
        
        MaxKernels = [];
        
        ExpKernels = [];
        LogKernels = [];
        AbsKernels = [];
    }
}