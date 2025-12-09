using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public static partial class Compute
{
    #region Properties & Fields
    public static ConcurrentDictionary<int, Accelerator> Accelerators { get; } = [];
    public static ConcurrentDictionary<string, int> AcceleratorIndices { get; } = [];
    public static ConcurrentDictionary<int, bool> InUse { get; } = [];
    public static Context Context { get; }

    private static readonly ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentStack<MemoryBuffer1D<float, Stride1D.Dense>>>> _pool = []; // _pool[aidx] -> stacks[size] -> buffers
    #endregion

    static Compute()
    {
        Context = Context.Create(b => b
            .Default()
            .EnableAlgorithms()
            .AllAccelerators());

        HashSet<(AcceleratorType, string, long)> seen = [];

        int aidx = 0;
        foreach (Device device in Context.Devices.Where(device => seen.Add((device.AcceleratorType, device.Name, device.MemorySize))))
        {
            Accelerators[aidx] = device.CreateAccelerator(Context);
            AcceleratorIndices[Accelerators[aidx].Name] = aidx;
            
            InUse[aidx] = false;
            _pool[aidx] = [];
            
            aidx++;
        }
        InitializeCuBlas();
        InitializeBootstrapKernels();
    }

    #region Management
    public static void ClearAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Clear(i);
    }
    public static void Clear(int aidx)
    {
        Synchronize(aidx);
        foreach (var kvp in _pool[aidx])
        {
            while (kvp.Value.TryPop(out var buffer)) buffer.Dispose();
            _pool[aidx].TryRemove(kvp);
        }
        _pool[aidx].Clear();
    }
    #region Synchronization
    public static void Synchronize(int aidx) => Accelerators[aidx].Synchronize();

    public static void SynchronizeAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Synchronize(i);
    }

    public static void Dispose()
    {
        GC.WaitForPendingFinalizers();
        ClearAll();
        Context.Dispose();
        foreach (var accelerator in Accelerators.Values) accelerator.Dispose();
        foreach (var blas in _cublasHandles.Values) blas?.Dispose();
        CleanupCuBlas();
    }
    #endregion
    #region Accelerator Management
    public static int RequestAccelerator(bool gpu = true)
    {
        Accelerator accelerator;
        var available = InUse.Values.Select((b, i) => (b, i)).Where(t => !t.b).Select((_, i) => i).ToList();
        
        if (available.Count == 0) throw new Exception("No accelerators available.");
        if (gpu) accelerator = Accelerators.Values.FirstOrDefault(a => a is CudaAccelerator) ?? Accelerators[available[0]];
        else accelerator = Accelerators[available[0]];
        
        var aidx = GetAcceleratorIndex(accelerator);
        InUse[aidx] = true;
        return aidx;
    }

    public static void ReleaseAccelerator(int aidx)
    {
        InUse[aidx] = false;
        Clear(aidx);
    }
    #endregion
    #endregion

    #region Returns
    public static void Return(MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        int aidx = buffer.AcceleratorIndex();
        int size = (int)buffer.Length;

        if (_pool[aidx].TryGetValue(size, out var stack)) stack.Push(buffer);
        else _pool[aidx][size] = new([buffer]);
    }
    public static void Return(params MemoryBuffer1D<float, Stride1D.Dense>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    } 
    #endregion
    #region Gets, Makes, & Encases
    public static MemoryBuffer1D<float, Stride1D.Dense> Get(int aidx, int size) => TryGetFrom(aidx, size);
    public static MemoryBuffer1D<float, Stride1D.Dense>[] Get(int aidx, int count, int size)
    {
        var result = new MemoryBuffer1D<float, Stride1D.Dense>[count];
        for (int i = 0; i < count; i++) result[i] = Get(aidx, size);
        return result;
    }
    public static MemoryBuffer1D<float, Stride1D.Dense> GetLike(MemoryBuffer1D<float, Stride1D.Dense> a) => Get(a.AcceleratorIndex(), (int)a.Length);

    public static MemoryBuffer1D<float, Stride1D.Dense> Make(int aidx, float[] values)
    {
        var result = Get(aidx, values.Length);
        result.CopyFromCPU(values);
        return result;
    }
    public static MemoryBuffer1D<float, Stride1D.Dense> Make(int aidx, int size, float value) => Make(aidx, Enumerable.Repeat(value, size).ToArray());

    public static Value<T> Encase<T>(Value<T> alike, Action<MemoryBuffer1D<float, Stride1D.Dense>> compute) where T : notnull
    {
        var result = GetLike(alike);
        compute(result);
        return alike.Create(result, alike.Shape);
    }
    public static MemoryBuffer1D<float, Stride1D.Dense> Encase(MemoryBuffer1D<float, Stride1D.Dense> alike, Action<MemoryBuffer1D<float, Stride1D.Dense>> compute)
    {
        var result = GetLike(alike);
        compute(result);
        return result;
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> Encase(int aidx, int size, Action<MemoryBuffer1D<float, Stride1D.Dense>> compute)
    {
        var result = Get(aidx, size);
        compute(result);
        return result;
    }
    #endregion
    #region Helpers
    public static int GetAcceleratorIndex(Accelerator accelerator) => AcceleratorIndices[accelerator.Name];
    public static bool IsGpuAccelerator(int aidx) => Accelerators[aidx] is CudaAccelerator;
    public static AcceleratorStream GetStream(int aidx) => Accelerators[aidx].DefaultStream;
    #region Calls & Call Overloads
    #region Calls
    public static void Call<T>(int aidx, Action<T>[] kernels, T value) =>
        kernels[aidx](value);
    public static void Call<T1, T2>(int aidx, Action<T1, T2>[] kernels, T1 a, T2 b) =>
        kernels[aidx](a, b);
    public static void Call<T1, T2, T3>(int aidx, Action<T1, T2, T3>[] kernels, T1 a, T2 b, T3 c) =>
        kernels[aidx](a, b, c);
    public static void Call<T1, T2, T3, T4>(int aidx, Action<T1, T2, T3, T4>[] kernels, T1 a, T2 b, T3 c, T4 d) =>
        kernels[aidx](a, b, c, d);
    public static void Call<T1, T2, T3, T4, T5>(int aidx, Action<T1, T2, T3, T4, T5>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e) =>
        kernels[aidx](a, b, c, d, e);
    public static void Call<T1, T2, T3, T4, T5, T6>(int aidx, Action<T1, T2, T3, T4, T5, T6>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f) =>
        kernels[aidx](a, b, c, d, e, f);
    public static void Call<T1, T2, T3, T4, T5, T6, T7>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g) =>
        kernels[aidx](a, b, c, d, e, f, g);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h) =>
        kernels[aidx](a, b, c, d, e, f, g, h);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i) => kernels[aidx](a, b, c, d, e, f, g, h, i);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j) => kernels[aidx](a, b, c, d, e, f, g, h, i, j);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k) => kernels[aidx](a, b, c, d, e, f, g, h, i, j, k);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k, T12 l) => kernels[aidx](a, b, c, d, e, f, g, h, i, j, k, l);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>[] kernels,
    T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k, T12 l, T13 m) => kernels[aidx](a, b, c, d, e, f, g, h, i, j, k, l, m);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>[] kernels,
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k, T12 l, T13 m, T14 n) => kernels[aidx](a, b, c, d, e, f, g, h, i, j, k, l, m, n);
    
    public static void Call(Action<Index1D, ArrayView1D<float, Stride1D.Dense>>[] kernels, ArrayView1D<float, Stride1D.Dense> view) => 
        kernels[view.AcceleratorIndex()](view.IntExtent, view);
    public static void Call<T>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T>[] kernels, ArrayView1D<float, Stride1D.Dense> a, T b) =>
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b);
    public static void Call<T1, T2>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2>[] kernels, ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c) => 
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c);
    public static void Call<T1, T2, T3>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3>[] kernels, ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d) => 
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d);
    public static void Call<T1, T2, T3, T4>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e);
    public static void Call<T1, T2, T3, T4, T5>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f);
    public static void Call<T1, T2, T3, T4, T5, T6>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g);
    public static void Call<T1, T2, T3, T4, T5, T6, T7>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j, k);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>[] kernels, ArrayView1D<float, Stride1D.Dense> a,
    T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l, T12 m) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l, m);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>[] kernels, ArrayView1D<float, Stride1D.Dense> a,
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l, T12 m, T13 n) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l, m, n);
    #endregion
    #region Binary Calls
    public static MemoryBuffer1D<float, Stride1D.Dense> BinaryCall(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a, 
        MemoryBuffer1D<float, Stride1D.Dense> b) => Encase(a, r => Call(kernels, r, a, b));

    public static Value<T> BinaryCall<T>(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels,
        Value<T> a, 
        Value<T> b) 
        where T : notnull => a.CreateAlike(BinaryCall(kernels, a.Data, b.Data));
    #endregion
    #region Unary Calls
    public static MemoryBuffer1D<float, Stride1D.Dense> UnaryCall(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a) => Encase(a, r => Call(kernels, r, a));
    
    public static Value<T> UnaryCall<T>(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels, 
        Value<T> a) 
        where T : notnull => a.CreateAlike(UnaryCall(kernels, a.Data));
    #endregion
    #region Call Chains
    public static void CallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial, 
        MemoryBuffer1D<float, Stride1D.Dense> result, 
        params Action<MemoryBuffer1D<float, Stride1D.Dense>>[] ops)
    {
        Call(CopyKernels, initial, result);
        foreach (var op in ops) op(result);
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> CallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial,
        params Action<MemoryBuffer1D<float, Stride1D.Dense>>[] ops) => Encase(initial, r => CallChain(initial, r, ops));
    #endregion
    #endregion
    #region Load & Load Overloads
    public static Action<Index1D, T>[] Load<T>(Action<Index1D, T> kernel)
        where T : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public static Action<Index1D, T1, T2>[] Load<T1, T2>(Action<Index1D, T1, T2> kernel)
        where T1 : struct
        where T2 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public static Action<Index1D, T1, T2, T3>[] Load<T1, T2, T3>(Action<Index1D, T1, T2, T3> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public static Action<Index1D, T1, T2, T3, T4>[] Load<T1, T2, T3, T4>(Action<Index1D, T1, T2, T3, T4> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public static Action<Index1D, T1, T2, T3, T4, T5>[] Load<T1, T2, T3, T4, T5>(Action<Index1D, T1, T2, T3, T4, T5> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public static Action<Index1D, T1, T2, T3, T4, T5, T6>[] Load<T1, T2, T3, T4, T5, T6>(Action<Index1D, T1, T2, T3, T4, T5, T6> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7>[] Load<T1, T2, T3, T4, T5, T6, T7>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8>[] Load<T1, T2, T3, T4, T5, T6, T7, T8>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct 
        where T8 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9>[] Load<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct 
        where T8 : struct
        where T9 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>[] Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct 
        where T8 : struct
        where T9 : struct
        where T10 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>[] Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct 
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>[] Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct 
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    
    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>[] Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct
        where T13 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();

    public static Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>[] Load<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>(Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
        where T11 : struct
        where T12 : struct
        where T13 : struct
        where T14 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();

    
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 1 view
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 2 views
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 3 views
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 4 views
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 5 views

    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 6 views
    
public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 7 views
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 8 views
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 9 views
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 10 views
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 11 views
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 12 views
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 13 views
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 14 views
    
    #endregion
    
    public static MemoryBuffer1D<float, Stride1D.Dense> Allocate(int aidx, int size) => Accelerators[aidx].Allocate1D<float>(size);
    private static MemoryBuffer1D<float, Stride1D.Dense> TryGetFrom(int aidx, int size)
    {
        MemoryBuffer1D<float, Stride1D.Dense> buffer;
        
        if (_pool[aidx].TryGetValue(size, out var stack))
            buffer = stack.TryPop(out var result) ? result : Allocate(aidx, size);
        else buffer = Allocate(aidx, size);

        Call(FillKernels, buffer, 0);
        return buffer;
    }
    #endregion
}