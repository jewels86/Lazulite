using ILGPU;
using ILGPU.IR.Transformations;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Lazulite.Kernels;

namespace Lazulite;

public static partial class Compute
{
    #region Properties
    public static List<Accelerator> Accelerators { get; } = [];
    public static List<int> Users { get; } = [];
    public static Context Context { get; private set; }
    public static bool AllowGpu { get; set; } = true;
    public static bool GpuInUse { get; private set; } = false;
    public static double Epsilon { get; set; } = 1e-10;

    private static readonly List<List<MemoryBuffer1D<double, Stride1D.Dense>>> _deferred = [];
    private static readonly List<Dictionary<int, Stack<MemoryBuffer1D<double, Stride1D.Dense>>>> _pool = []; // aidx -> stack of buffers sorted by size
    // buffers in pool are NOT zeroed yet
    #endregion

    static Compute()
    {
        Context = Context.CreateDefault();
        RefreshDevices();
        InitializeKernels();
        WarmupKernelsAsync();
    }

    #region Management
    public static void RefreshDevices()
    {
        FlushAll();
        foreach (Accelerator accelerator in Accelerators) accelerator.Dispose();
        
        Accelerators.Clear();
        Context.Dispose();
        Context = Context.Create(b => b
            .Default()
            .EnableAlgorithms()
            .AllAccelerators());
        
        foreach (Device device in Context.Devices.Where(device => device is not CudaDevice || AllowGpu))
        {
            Accelerators.Add(device.CreateAccelerator(Context));
            Users.Add(0);
            _pool.Add([]);
            _deferred.Add([]);
            GpuInUse = true;
        }
    }
    public static void FlushAll()
    {
        foreach (MemoryBuffer1D<double, Stride1D.Dense> b in _deferred.SelectMany(buffer => buffer)) Return(b);
        foreach (Stack<MemoryBuffer1D<double, Stride1D.Dense>> stack in _pool.SelectMany(pool => pool.Values))
            while (stack.Count > 0)
                stack.Pop().Dispose();
        _deferred.Clear();
        _pool.Clear();
    }
    #region Synchronization
    public static void Synchronize(int aidx)
    {
        Accelerators[aidx].Synchronize();
        foreach (var deferred in _deferred[aidx]) Return(deferred);
    }
    public static void SynchronizeAll()
    {
        for (int i = 0; i < Accelerators.Count; i++) Synchronize(i);
    }
    #endregion
    #region Accelerator Management
    public static int RequestAccelerator(bool gpu = true)
    {
        Accelerator accelerator;
        var lowestUsers = Users.IndexOf(Users.Min());
        if (gpu) accelerator = Accelerators.FirstOrDefault(a => a is CudaAccelerator) ?? Accelerators[lowestUsers];
        else accelerator = Accelerators[lowestUsers];
        var aidx = GetAcceleratorIndex(accelerator);
        Users[aidx]++;
        return aidx;
    }

    public static void ReleaseAccelerator(int aidx) => Users[aidx]--;
    #endregion
    #endregion

    #region Returns
    public static void Return(MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        if (!GpuInUse) buffer.Dispose();
        int size = (int)buffer.Length;
        if (_pool[GetAcceleratorIndex(buffer.Accelerator)].TryGetValue(size, out var stack)) stack.Push(buffer);
        else _pool[GetAcceleratorIndex(buffer.Accelerator)][size] = new Stack<MemoryBuffer1D<double, Stride1D.Dense>>([buffer]);
    }
    public static void Return(params MemoryBuffer1D<double, Stride1D.Dense>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    } 
    public static void DeferReturn(MemoryBuffer1D<double, Stride1D.Dense> buffer) => _deferred[buffer.AcceleratorIndex()].Add(buffer);
    public static void DeferReturn(params MemoryBuffer1D<double, Stride1D.Dense>[] buffers) => _deferred[buffers[0].AcceleratorIndex()].AddRange(buffers);
    #endregion
    #region Gets
    public static MemoryBuffer1D<double, Stride1D.Dense> Get(int aidx, int size) => TryGetFrom(aidx, size);
    public static MemoryBuffer1D<double, Stride1D.Dense> GetTemp(int aidx, int size) => TryGetFrom(aidx, size).Defer();
    public static MemoryBuffer1D<double, Stride1D.Dense>[] Get(int aidx, int count, int size) => Enumerable.Range(0, count).Select(_ => Get(aidx, size)).ToArray();
    public static MemoryBuffer1D<double, Stride1D.Dense>[] GetTemps(int aidx, int count, int size) => Enumerable.Range(0, count).Select(_ => GetTemp(aidx, size)).ToArray();
    public static MemoryBuffer1D<double, Stride1D.Dense> GetLike(MemoryBuffer1D<double, Stride1D.Dense> a) => Get(a.AcceleratorIndex(), (int)a.Length);
    public static MemoryBuffer1D<double, Stride1D.Dense> GetTempLike(MemoryBuffer1D<double, Stride1D.Dense> a) => GetTemp(a.AcceleratorIndex(), (int)a.Length);
    
    public static Value<T> CreateLike<T>(Value<T> a) where T : notnull => a.Create(Get(a.AcceleratorIndex, a.TotalSize));
    public static Value<T> CreateTempLike<T>(Value<T> a) where T : notnull => a.Create(GetTemp(a.AcceleratorIndex, a.TotalSize));
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
    public static void Call(int aidx, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>>> kernels, ArrayView1D<double, Stride1D.Dense> view) => 
        kernels[aidx](view.IntExtent, view);
    public static void Call<T>(
        int aidx, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, T>> kernels,
        ArrayView1D<double, Stride1D.Dense> a, T b) => 
        kernels[aidx](a.IntExtent, a, b);
    public static void Call<T1, T2>(int aidx, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, T1, T2>> kernels,
        ArrayView1D<double, Stride1D.Dense> a, T1 b, T2 c) => 
        kernels[aidx](a.IntExtent, a, b, c);
    public static void Call<T1, T2, T3>(int aidx, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, T1, T2, T3>> kernels,
        ArrayView1D<double, Stride1D.Dense> a, T1 b, T2 c, T3 d) => 
        kernels[aidx](a.IntExtent, a, b, c, d);
    #endregion
    
    #region Binary Calls
    public static void BinaryCall(List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<double, Stride1D.Dense> a, MemoryBuffer1D<double, Stride1D.Dense> b, MemoryBuffer1D<double, Stride1D.Dense> result) =>
        kernels[a.AcceleratorIndex()](a.IntExtent, a, b, result);
    public static MemoryBuffer1D<double, Stride1D.Dense> BinaryCall(
        List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<double, Stride1D.Dense> a, MemoryBuffer1D<double, Stride1D.Dense> b)
    {
        var result = GetLike(a);
        BinaryCall(kernels, a, b, result);
        return result;
    }

    public static Value<T> BinaryCall<T>(
        List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> kernels,
        Value<T> a, Value<T> b) where T : notnull => a.Create(BinaryCall(kernels, a.Data, b.Data));

    public static void BinaryCallChain(
        MemoryBuffer1D<double, Stride1D.Dense> initial, MemoryBuffer1D<double, Stride1D.Dense> result,
        params (
            List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>>kernels,
            MemoryBuffer1D<double, Stride1D.Dense> operand)[] ops)
    {
        var aidx = initial.AcceleratorIndex();
        ops[0].kernels[aidx](initial.IntExtent, initial, ops[0].operand, result);
        for (int i = 1; i < ops.Length; i++) ops[i].kernels[aidx](result.IntExtent, result, ops[i].operand, result);
    }

    public static MemoryBuffer1D<double, Stride1D.Dense> BinaryCallChain(
        MemoryBuffer1D<double, Stride1D.Dense> initial,
        params (
            List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>>kernels,
            MemoryBuffer1D<double, Stride1D.Dense> operand)[] ops)
    {
        var result = GetLike(initial);
        BinaryCallChain(initial, result, ops);
        return result;
    }
    public static Value<T> BinaryCallChain<T>(
        Value<T> initial,
        params (
            List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>>kernels,
            Value<T> operand)[] ops) where T : notnull => initial.Create(BinaryCallChain(initial.Data, ops.Select(op => (op.kernels, op.operand.Data)).ToArray()));
    #endregion
    
    #region Unary Calls
    public static void UnaryCall(List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<double, Stride1D.Dense> a, MemoryBuffer1D<double, Stride1D.Dense> result) =>
        kernels[a.AcceleratorIndex()](a.IntExtent, a, result);

    public static MemoryBuffer1D<double, Stride1D.Dense> UnaryCallUnaryCall(List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<double, Stride1D.Dense> a)
    {
        var result = GetLike(a);
        UnaryCall(kernels, a, result);
        return result;
    }
    
    public static Value<T> UnaryCall<T>(
        List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> kernels,
        Value<T> a) where T : notnull => a.Create(UnaryCallUnaryCall(kernels, a.Data));
    
    
    #endregion
    #endregion
    
    public static MemoryBuffer1D<double, Stride1D.Dense> Allocate(int aidx, int size) => Accelerators[aidx].Allocate1D<double>(size);
    private static MemoryBuffer1D<double, Stride1D.Dense> TryGetFrom(int aidx, int size)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive", nameof(size));
        var pool = _pool[aidx];
        MemoryBuffer1D<double, Stride1D.Dense> buffer;
        
        if (pool.TryGetValue(size, out var stack) && stack.Count > 0) buffer = stack.Pop();
        else buffer = Allocate(aidx, size);
        
        Call(buffer.AcceleratorIndex(), FillKernels, buffer.View, 0.0);
        return buffer;
    }
    #endregion
}