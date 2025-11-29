using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public sealed partial class Compute : IDisposable
{
    #region Properties & Fields
    public ConcurrentDictionary<int, Accelerator> Accelerators { get; } = [];
    public ConcurrentDictionary<string, int> AcceleratorIndices { get; } = [];
    public ConcurrentDictionary<int, bool> InUse { get; } = [];
    public Context Context { get; private set; }
    public static Compute Instance => _lazyInstance.Value;

    private readonly ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentStack<MemoryBuffer1D<float, Stride1D.Dense>>>> _pool = []; // _pool[aidx] -> stacks[size] -> buffers
    private static readonly Lazy<Compute> _lazyInstance = new(() => new Compute());
    #endregion

    private Compute()
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
    public void ClearAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Clear(i);
    }
    public void Clear(int aidx)
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
    public void Synchronize(int aidx) => Accelerators[aidx].Synchronize();

    public void SynchronizeAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Synchronize(i);
    }

    public void Dispose()
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
    public int RequestAccelerator(bool gpu = true)
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

    public void ReleaseAccelerator(int aidx)
    {
        InUse[aidx] = false;
        Clear(aidx);
    }
    #endregion
    #endregion

    #region Returns
    public void Return(MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        int aidx = buffer.AcceleratorIndex();
        int size = (int)buffer.Length;
        
        if (_pool[aidx].TryGetValue(size, out var stack))
        {
            if (stack.Contains(buffer)) throw new Exception("Buffer already returned.");
            stack.Push(buffer);
        }
        else _pool[aidx][size] = new([buffer]);
    }
    public void Return(params MemoryBuffer1D<float, Stride1D.Dense>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    } 
    #endregion
    #region Gets, Makes, & Encases
    public MemoryBuffer1D<float, Stride1D.Dense> Get(int aidx, int size) => TryGetFrom(aidx, size);
    public MemoryBuffer1D<float, Stride1D.Dense>[] Get(int aidx, int count, int size)
    {
        var result = new MemoryBuffer1D<float, Stride1D.Dense>[count];
        for (int i = 0; i < count; i++) result[i] = Get(aidx, size);
        return result;
    }
    public MemoryBuffer1D<float, Stride1D.Dense> GetLike(MemoryBuffer1D<float, Stride1D.Dense> a) => Get(a.AcceleratorIndex(), (int)a.Length);

    public MemoryBuffer1D<float, Stride1D.Dense> Make(int aidx, float[] values)
    {
        var result = Get(aidx, values.Length);
        result.CopyFromCPU(values);
        return result;
    }
    public MemoryBuffer1D<float, Stride1D.Dense> Make(int aidx, int size, float value) => Make(aidx, Enumerable.Repeat(value, size).ToArray());
    
    public Value<T> CreateLike<T>(Value<T> a) where T : notnull => a.Create(Get(a.AcceleratorIndex, a.TotalSize), a.Shape);

    public Value<T> Encase<T>(Value<T> alike, Action<MemoryBuffer1D<float, Stride1D.Dense>> compute) where T : notnull
    {
        var result = GetLike(alike);
        compute(result);
        return alike.Create(result, alike.Shape);
    }
    public MemoryBuffer1D<float, Stride1D.Dense> Encase(MemoryBuffer1D<float, Stride1D.Dense> alike, Action<MemoryBuffer1D<float, Stride1D.Dense>> compute)
    {
        var result = GetLike(alike);
        compute(result);
        return result;
    }
    public MemoryBuffer1D<float, Stride1D.Dense> Encase(int aidx, int size, Action<MemoryBuffer1D<float, Stride1D.Dense>> compute) => Encase(Get(aidx, size), compute);
    #endregion
    #region Helpers
    public int GetAcceleratorIndex(Accelerator accelerator) => AcceleratorIndices[accelerator.Name];
    public bool IsGpuAccelerator(int aidx) => Accelerators[aidx] is CudaAccelerator;
    public AcceleratorStream GetStream(int aidx) => Accelerators[aidx].DefaultStream;
    public AcceleratorStream GetStream(Accelerator accelerator) => accelerator.DefaultStream;
    
    #region Calls & Call Overloads
    #region Calls
    public void Call<T>(int aidx, Action<T>[] kernels, T value) =>
        kernels[aidx](value);
    public void Call<T1, T2>(int aidx, Action<T1, T2>[] kernels, T1 a, T2 b) =>
        kernels[aidx](a, b);
    public void Call<T1, T2, T3>(int aidx, Action<T1, T2, T3>[] kernels, T1 a, T2 b, T3 c) =>
        kernels[aidx](a, b, c);
    public void Call<T1, T2, T3, T4>(int aidx, Action<T1, T2, T3, T4>[] kernels, T1 a, T2 b, T3 c, T4 d) =>
        kernels[aidx](a, b, c, d);
    public void Call<T1, T2, T3, T4, T5>(int aidx, Action<T1, T2, T3, T4, T5>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e) =>
        kernels[aidx](a, b, c, d, e);
    public void Call<T1, T2, T3, T4, T5, T6>(int aidx, Action<T1, T2, T3, T4, T5, T6>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f) =>
        kernels[aidx](a, b, c, d, e, f);
    public void Call<T1, T2, T3, T4, T5, T6, T7>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g) =>
        kernels[aidx](a, b, c, d, e, f, g);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8>[] kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h) =>
        kernels[aidx](a, b, c, d, e, f, g, h);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i) => kernels[aidx](a, b, c, d, e, f, g, h, i);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j) => kernels[aidx](a, b, c, d, e, f, g, h, i, j);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k) => kernels[aidx](a, b, c, d, e, f, g, h, i, j, k);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(int aidx, Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>[] kernels, 
        T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k, T12 l) => kernels[aidx](a, b, c, d, e, f, g, h, i, j, k, l);
    
    public void Call(Action<Index1D, ArrayView1D<float, Stride1D.Dense>>[] kernels, ArrayView1D<float, Stride1D.Dense> view) => 
        kernels[view.AcceleratorIndex()](view.IntExtent, view);
    public void Call<T>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T>[] kernels, ArrayView1D<float, Stride1D.Dense> a, T b) =>
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b);
    public void Call<T1, T2>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2>[] kernels, ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c) => 
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c);
    public void Call<T1, T2, T3>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3>[] kernels, ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d) => 
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d);
    public void Call<T1, T2, T3, T4>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e);
    public void Call<T1, T2, T3, T4, T5>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f);
    public void Call<T1, T2, T3, T4, T5, T6>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g);
    public void Call<T1, T2, T3, T4, T5, T6, T7>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j, k);
    public void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>[] kernels, ArrayView1D<float, Stride1D.Dense> a, 
        T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l) => kernels[a.AcceleratorIndex()](a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l);
    #endregion
    #region Binary Calls
    public MemoryBuffer1D<float, Stride1D.Dense> BinaryCall(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a, 
        MemoryBuffer1D<float, Stride1D.Dense> b)
    {
        var result = GetLike(a);
        Call(kernels, a, b, result);
        return result;
    }

    public Value<T> BinaryCall<T>(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels,
        Value<T> a, 
        Value<T> b) 
        where T : notnull => a.Create(BinaryCall(kernels, a.Data, b.Data), a.Shape);

    public void BinaryCallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial, 
        MemoryBuffer1D<float, Stride1D.Dense> result,
        params (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels, MemoryBuffer1D<float, Stride1D.Dense> operand)[] ops)
    {
        for (int i = 0; i < ops.Length; i++) Call(ops[i].kernels, result, ops[i].operand, result);
    }

    public MemoryBuffer1D<float, Stride1D.Dense> BinaryCallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial,
        params (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels, MemoryBuffer1D<float, Stride1D.Dense> operand)[] ops)
    {
        var result = GetLike(initial);
        BinaryCallChain(initial, result, ops);
        return result;
    }
    public Value<T> BinaryCallChain<T>(
        Value<T> initial,
        params (Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels, Value<T> operand)[] ops) 
        where T : notnull => initial.Create(BinaryCallChain(initial.Data, ops.Select(op => (op.kernels, op.operand.Data)).ToArray()), initial.Shape);
    #endregion
    #region Unary Calls
    public MemoryBuffer1D<float, Stride1D.Dense> UnaryCall(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a)
    {
        var result = GetLike(a);
        Call(kernels, a, result);
        return result;
    }
    
    public Value<T> UnaryCall<T>(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] kernels, 
        Value<T> a) 
        where T : notnull => a.Create(UnaryCall(kernels, a.Data), a.Shape);
    #endregion
    #region Call Chains
    public void CallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial, 
        MemoryBuffer1D<float, Stride1D.Dense> result, 
        params Action<MemoryBuffer1D<float, Stride1D.Dense>>[] ops)
    {
        Call(CopyKernels, initial, result);
        foreach (var op in ops) op(result);
    }

    public MemoryBuffer1D<float, Stride1D.Dense> CallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial,
        params Action<MemoryBuffer1D<float, Stride1D.Dense>>[] ops)
    {
        var result = GetLike(initial);
        CallChain(initial, result, ops);
        return result;
    }
    #endregion
    #endregion
    #region Load & Load Overloads
    public Action<Index1D, T>[] Load<T>(Action<Index1D, T> kernel)
        where T : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public Action<Index1D, T1, T2>[] Load<T1, T2>(Action<Index1D, T1, T2> kernel)
        where T1 : struct
        where T2 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public Action<Index1D, T1, T2, T3>[] Load<T1, T2, T3>(Action<Index1D, T1, T2, T3> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public Action<Index1D, T1, T2, T3, T4>[] Load<T1, T2, T3, T4>(Action<Index1D, T1, T2, T3, T4> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public Action<Index1D, T1, T2, T3, T4, T5>[] Load<T1, T2, T3, T4, T5>(Action<Index1D, T1, T2, T3, T4, T5> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    public Action<Index1D, T1, T2, T3, T4, T5, T6>[] Load<T1, T2, T3, T4, T5, T6>(Action<Index1D, T1, T2, T3, T4, T5, T6> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray();
    
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>>[] Load(Action<Index1D, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 1 view
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 2 views
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 3 views
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 4 views
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 5 views
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Values.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToArray(); // 6 views
    #endregion
    
    public MemoryBuffer1D<float, Stride1D.Dense> Allocate(int aidx, int size) => Accelerators[aidx].Allocate1D<float>(size);
    private MemoryBuffer1D<float, Stride1D.Dense> TryGetFrom(int aidx, int size)
    {
        MemoryBuffer1D<float, Stride1D.Dense> buffer;
        
        if (_pool[aidx].TryGetValue(size, out var stack))
            buffer = stack.TryPop(out var result) ? result : Allocate(aidx, size);
        else buffer = Allocate(aidx, size);

        Call(aidx, FillKernels, buffer.IntExtent, buffer, 0);
        return buffer;
    }
    #endregion
}