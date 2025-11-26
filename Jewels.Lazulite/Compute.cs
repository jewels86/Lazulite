using System.Linq.Expressions;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.IR.Transformations;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Jewels.Lazulite.Kernels;

namespace Jewels.Lazulite;

public static partial class Compute
{
    #region Properties
    public static List<Accelerator> Accelerators { get; } = [];
    public static List<bool> InUse { get; } = [];
    public static Context Context { get; private set; }
    public static bool AllowGpu { get; set; } = true;
    public static bool GpuInUse { get; private set; } = false;

    private static readonly List<List<MemoryBuffer1D<float, Stride1D.Dense>>> _deferred = [];
    private static readonly List<Dictionary<int, Stack<MemoryBuffer1D<float, Stride1D.Dense>>>> _pool = []; // aidx -> stack of buffers sorted by size
    #endregion

    static Compute()
    {
        Context = Context.CreateDefault();
        RefreshDevices();
        InitializeCoreKernels();
    }

    #region Management
    public static void RefreshDevices()
    {
        ClearAll();
        foreach (Accelerator accelerator in Accelerators) accelerator.Dispose();
        
        Accelerators.Clear();
        Context.Dispose();
        Context = Context.Create(b => b
            .Default()
            .EnableAlgorithms()
            .AllAccelerators());

        HashSet<(AcceleratorType, string, long)> seen = [];

        int aidx = 0;
        foreach (Device device in Context.Devices.Where(device => device is not CudaDevice || AllowGpu))
        {
            if (!seen.Add((device.AcceleratorType, device.Name, device.MemorySize))) continue;
            Accelerators.Add(device.CreateAccelerator(Context));
            InUse.Add(false);
            _pool.Add([]);
            _deferred.Add([]);
            if (device is CudaDevice)
            {
                GpuInUse = true;
                GetCuBlas(aidx);
            }
            aidx++;
        }
    }
    public static void ClearAll()
    {
        foreach (MemoryBuffer1D<float, Stride1D.Dense> b in _deferred.SelectMany(buffer => buffer)) Return(b);
        foreach (Stack<MemoryBuffer1D<float, Stride1D.Dense>> stack in _pool.SelectMany(pool => pool.Values))
            while (stack.Count > 0)
                stack.Pop().Dispose();
        foreach (var deferred in _deferred) deferred.Clear();
        foreach (var pool in _pool) pool.Clear();
        CleanupCuBlas();
    }
    #region Synchronization
    public static void Synchronize(int aidx)
    {
        Flush(aidx);
        Accelerators[aidx].Synchronize();
    }

    public static void SynchronizeAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Synchronize(i);
    }

    public static void Flush(int aidx)
    {
        foreach (var deferred in _deferred[aidx]) Return(deferred);
        _deferred[aidx].Clear();
    }

    public static void FlushAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Flush(i);
    }
    #endregion
    #region Accelerator Management
    public static int RequestAccelerator(bool gpu = true)
    {
        Accelerator accelerator;
        var available = InUse.Select((b, i) => (b, i)).Where(t => !t.b).Select((b, i) => i).ToList();
        
        if (available.Count == 0) throw new Exception("No accelerators available.");
        if (gpu) accelerator = Accelerators.FirstOrDefault(a => a is CudaAccelerator) ?? Accelerators[available[0]];
        else accelerator = Accelerators[available[0]];
        
        var aidx = GetAcceleratorIndex(accelerator);
        InUse[aidx] = true;
        return aidx;
    }

    public static void ReleaseAccelerator(int aidx) => InUse[aidx] = false;
    #endregion
    #endregion

    #region Returns
    public static void Return(MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        if (!GpuInUse || !IsGpuAccelerator(buffer.AcceleratorIndex())) { buffer.Dispose(); return; }
        int size = (int)buffer.Length;
        if (_pool[GetAcceleratorIndex(buffer.Accelerator)].TryGetValue(size, out var stack)) stack.Push(buffer);
        else _pool[GetAcceleratorIndex(buffer.Accelerator)][size] = new Stack<MemoryBuffer1D<float, Stride1D.Dense>>([buffer]);
    }
    public static void Return(params MemoryBuffer1D<float, Stride1D.Dense>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    } 
    public static void DeferReturn(MemoryBuffer1D<float, Stride1D.Dense> buffer) => _deferred[buffer.AcceleratorIndex()].Add(buffer);
    public static void DeferReturn(params MemoryBuffer1D<float, Stride1D.Dense>[] buffers) => _deferred[buffers[0].AcceleratorIndex()].AddRange(buffers);
    #endregion
    #region Gets
    public static MemoryBuffer1D<float, Stride1D.Dense> Get(int aidx, int size) => TryGetFrom(aidx, size);
    public static MemoryBuffer1D<float, Stride1D.Dense> GetTemp(int aidx, int size) => TryGetFrom(aidx, size).DeferReturn();
    public static MemoryBuffer1D<float, Stride1D.Dense>[] Get(int aidx, int count, int size) => Enumerable.Range(0, count).Select(_ => Get(aidx, size)).ToArray();
    public static MemoryBuffer1D<float, Stride1D.Dense>[] GetTemps(int aidx, int count, int size) => Enumerable.Range(0, count).Select(_ => GetTemp(aidx, size)).ToArray();
    public static MemoryBuffer1D<float, Stride1D.Dense> GetLike(MemoryBuffer1D<float, Stride1D.Dense> a) => Get(a.AcceleratorIndex(), (int)a.Length);
    public static MemoryBuffer1D<float, Stride1D.Dense> GetTempLike(MemoryBuffer1D<float, Stride1D.Dense> a) => GetTemp(a.AcceleratorIndex(), (int)a.Length);

    public static MemoryBuffer1D<float, Stride1D.Dense> Make(int aidx, float[] values)
    {
        var result = Get(aidx, values.Length);
        result.CopyFromCPU(values);
        return result;
    }
    public static MemoryBuffer1D<float, Stride1D.Dense> MakeTemp(int aidx, float[] values)
    {
        var result = GetTemp(aidx, values.Length);
        result.CopyFromCPU(values);
        return result;
    }
    public static MemoryBuffer1D<float, Stride1D.Dense> Make(int aidx, int size, float value) => Make(aidx, Enumerable.Repeat(value, size).ToArray());
    public static MemoryBuffer1D<float, Stride1D.Dense> MakeTemp(int aidx, int size, float value) => MakeTemp(aidx, Enumerable.Repeat(value, size).ToArray());
    
    public static Value<T> CreateLike<T>(Value<T> a) where T : notnull => a.Create(Get(a.AcceleratorIndex, a.TotalSize), a.Shape);
    public static Value<T> CreateTempLike<T>(Value<T> a) where T : notnull => a.Create(GetTemp(a.AcceleratorIndex, a.TotalSize), a.Shape);

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
    public static MemoryBuffer1D<float, Stride1D.Dense> Encase(int aidx, int size, Action<MemoryBuffer1D<float, Stride1D.Dense>> compute) => Encase(Get(aidx, size), compute);
    #endregion
    #region Helpers
    public static int GetAcceleratorIndex(Accelerator accelerator) => Accelerators.IndexOf(accelerator);
    public static bool IsGpuAccelerator(int aidx) => Accelerators[aidx] is CudaAccelerator;
    public static AcceleratorStream GetStream(int aidx) => Accelerators[aidx].DefaultStream;
    public static AcceleratorStream GetStream(Accelerator accelerator) => accelerator.DefaultStream;
    
    #region Calls & Call Overloads
    #region Calls
    public static void Call<T>(int aidx, List<Action<T>> kernels, T value) =>
        kernels[aidx](value);
    public static void Call<T1, T2>(int aidx, List<Action<T1, T2>> kernels, T1 a, T2 b) =>
        kernels[aidx](a, b);
    public static void Call<T1, T2, T3>(int aidx, List<Action<T1, T2, T3>> kernels, T1 a, T2 b, T3 c) =>
        kernels[aidx](a, b, c);
    public static void Call<T1, T2, T3, T4>(int aidx, List<Action<T1, T2, T3, T4>> kernels, T1 a, T2 b, T3 c, T4 d) =>
        kernels[aidx](a, b, c, d);
    public static void Call<T1, T2, T3, T4, T5>(int aidx, List<Action<T1, T2, T3, T4, T5>> kernels, T1 a, T2 b, T3 c, T4 d, T5 e) =>
        kernels[aidx](a, b, c, d, e);
    public static void Call<T1, T2, T3, T4, T5, T6>(int aidx, List<Action<T1, T2, T3, T4, T5, T6>> kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f) =>
        kernels[aidx](a, b, c, d, e, f);
    public static void Call<T1, T2, T3, T4, T5, T6, T7>(int aidx, List<Action<T1, T2, T3, T4, T5, T6, T7>> kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g) =>
        kernels[aidx](a, b, c, d, e, f, g);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8>(int aidx, List<Action<T1, T2, T3, T4, T5, T6, T7, T8>> kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h) =>
        kernels[aidx](a, b, c, d, e, f, g, h);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(int aidx, List<Action<T1, T2, T3, T4, T5, T6, T7, T8, T9>> kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i) =>
        kernels[aidx](a, b, c, d, e, f, g, h, i);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(int aidx, List<Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>> kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j) =>
        kernels[aidx](a, b, c, d, e, f, g, h, i, j);
    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(int aidx, List<Action<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>> kernels, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 i, T10 j, T11 k) =>
        kernels[aidx](a, b, c, d, e, f, g, h, i, j, k);
    public static void Call(int aidx, List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>>> kernels, ArrayView1D<float, Stride1D.Dense> view) => 
        kernels[aidx](view.IntExtent, view);
    public static void Call<T>(
        int aidx, List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T>> kernels,
        ArrayView1D<float, Stride1D.Dense> a, T b) => 
        kernels[aidx](a.IntExtent, a, b);
    public static void Call<T1, T2>(int aidx, List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2>> kernels,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c) => 
        kernels[aidx](a.IntExtent, a, b, c);
    public static void Call<T1, T2, T3>(int aidx, List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3>> kernels,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d) => 
        kernels[aidx](a.IntExtent, a, b, c, d);
    public static void Call<T1, T2, T3, T4>(int aidx, List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4>> kernels,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e) => 
        kernels[aidx](a.IntExtent, a, b, c, d, e);
    public static void Call<T1, T2, T3, T4, T5>(int aidx, List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5>> kernels,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f) => 
        kernels[aidx](a.IntExtent, a, b, c, d, e, f);
    #endregion
    
    #region Binary Calls
    public static void BinaryCall(List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b, MemoryBuffer1D<float, Stride1D.Dense> result) =>
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b, result);
    public static MemoryBuffer1D<float, Stride1D.Dense> BinaryCall(
        List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b)
    {
        var result = GetLike(a);
        BinaryCall(kernels, a, b, result);
        return result;
    }

    public static Value<T> BinaryCall<T>(
        List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels,
        Value<T> a, Value<T> b) where T : notnull => a.Create(BinaryCall(kernels, a.Data, b.Data), a.Shape);

    public static void BinaryCallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial, MemoryBuffer1D<float, Stride1D.Dense> result,
        params (
            List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>>kernels,
            MemoryBuffer1D<float, Stride1D.Dense> operand)[] ops)
    {
        var aidx = initial.AcceleratorIndex();
        ops[0].kernels[aidx](initial.IntExtent, initial, ops[0].operand, result);
        for (int i = 1; i < ops.Length; i++) ops[i].kernels[aidx](result.IntExtent, result, ops[i].operand, result);
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> BinaryCallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial,
        params (
            List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>>kernels,
            MemoryBuffer1D<float, Stride1D.Dense> operand)[] ops)
    {
        var result = GetLike(initial);
        BinaryCallChain(initial, result, ops);
        return result;
    }
    public static Value<T> BinaryCallChain<T>(
        Value<T> initial,
        params (
            List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>>kernels,
            Value<T> operand)[] ops) where T : notnull => 
        initial.Create(BinaryCallChain(initial.Data, ops.Select(op => (op.kernels, op.operand.Data)).ToArray()), initial.Shape);
    #endregion
    
    #region Unary Calls
    public static void UnaryCall(List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> result) =>
        kernels[a.AcceleratorIndex()](a.IntExtent, a, result);

    public static MemoryBuffer1D<float, Stride1D.Dense> UnaryCallUnaryCall(List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a)
    {
        var result = GetLike(a);
        UnaryCall(kernels, a, result);
        return result;
    }
    
    public static Value<T> UnaryCall<T>(
        List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels,
        Value<T> a) where T : notnull => a.Create(UnaryCallUnaryCall(kernels, a.Data), a.Shape);
    #endregion
    #endregion
    #region Load & Load Overloads
    public static List<Action<Index1D, T>> Load<T>(Action<Index1D, T> kernel)
        where T : struct =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, T1, T2>> Load<T1, T2>(Action<Index1D, T1, T2> kernel)
        where T1 : struct
        where T2 : struct =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, T1, T2, T3>> Load<T1, T2, T3>(Action<Index1D, T1, T2, T3> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, T1, T2, T3, T4>> Load<T1, T2, T3, T4>(Action<Index1D, T1, T2, T3, T4> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, T1, T2, T3, T4, T5>> Load<T1, T2, T3, T4, T5>(Action<Index1D, T1, T2, T3, T4, T5> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, T1, T2, T3, T4, T5, T6>> Load<T1, T2, T3, T4, T5, T6>(Action<Index1D, T1, T2, T3, T4, T5, T6> kernel)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    
    public static List<Action<Index1D,ArrayView1D<float, Stride1D.Dense>>> Load(Action<Index1D, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) => 
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> Load(
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> kernel) =>
        Accelerators.Select(a => a.LoadAutoGroupedStreamKernel(kernel)).ToList();
    #endregion
    
    public static MemoryBuffer1D<float, Stride1D.Dense> Allocate(int aidx, int size) => Accelerators[aidx].Allocate1D<float>(size);
    private static MemoryBuffer1D<float, Stride1D.Dense> TryGetFrom(int aidx, int size)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive", nameof(size));
        var pool = _pool[aidx];
        MemoryBuffer1D<float, Stride1D.Dense> buffer;
        
        if (pool.TryGetValue(size, out var stack) && stack.Count > 0 && !IsGpuAccelerator(aidx)) buffer = stack.Pop();
        else buffer = Allocate(aidx, size);
        
        Call(buffer.AcceleratorIndex(), FillKernels, buffer.View, 0);
        return buffer;
    }
    #endregion
    
}

public class ComputeScope : IDisposable
{
    public void Dispose() => Compute.ClearAll();
}