using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Lazulite.Kernels;

namespace Lazulite;

public static partial class Compute
{
    public static List<Accelerator> Accelerators { get; } = [];
    public static List<int> Users { get; } = [];
    public static Context Context { get; private set; }
    public static bool AllowGpu { get; set; } = true;
    public static bool GpuInUse { get; private set; } = false;

    private static readonly List<List<MemoryBuffer1D<double, Stride1D.Dense>>> _deferred = [];
    private static readonly List<Dictionary<int, Stack<MemoryBuffer1D<double, Stride1D.Dense>>>> _pool = []; // acceleratorIndex -> stack of buffers sorted by size
    // buffers in pool are NOT zeroed yet

    static Compute()
    {
        Context = Context.CreateDefault();
        RefreshDevices();
        InitializeKernels();
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
    public static void Synchronize(int acceleratorIndex)
    {
        Accelerators[acceleratorIndex].Synchronize();
        foreach (var deferred in _deferred[acceleratorIndex]) Return(deferred);
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

    public static void ReleaseAccelerator(int acceleratorIndex) => Users[acceleratorIndex]--;
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
    public static MemoryBuffer1D<double, Stride1D.Dense> Get(int acceleratorIndex, int size) => TryGetFrom(acceleratorIndex, size);
    public static MemoryBuffer1D<double, Stride1D.Dense> GetTemp(int acceleratorIndex, int size) => TryGetFrom(acceleratorIndex, size).Defer();
    public static MemoryBuffer1D<double, Stride1D.Dense>[] Get(int acceleratorIndex, int count, int size) => Enumerable.Range(0, count).Select(_ => Get(acceleratorIndex, size)).ToArray();
    public static MemoryBuffer1D<double, Stride1D.Dense>[] GetTemps(int acceleratorIndex, int count, int size) => Enumerable.Range(0, count).Select(_ => GetTemp(acceleratorIndex, size)).ToArray();

    #endregion
    #region Helpers
    public static int GetAcceleratorIndex(Accelerator accelerator) => Accelerators.IndexOf(accelerator);
    public static bool IsGpuAccelerator(int acceleratorIndex) => Accelerators[acceleratorIndex] is CudaAccelerator;
    public static AcceleratorStream GetStream(int acceleratorIndex) => Accelerators[acceleratorIndex].DefaultStream;
    public static AcceleratorStream GetStream(Accelerator accelerator) => accelerator.DefaultStream;
    
    #region Call Overloads
    public static void Call<T>(int acceleratorIndex, List<Action<T>> kernels, T value) =>
        kernels[acceleratorIndex](value);
    public static void Call<T1, T2>(int acceleratorIndex, List<Action<T1, T2>> kernels, T1 a, T2 b) =>
        kernels[acceleratorIndex](a, b);
    public static void Call<T1, T2, T3>(int acceleratorIndex, List<Action<T1, T2, T3>> kernels, T1 a, T2 b, T3 c) =>
        kernels[acceleratorIndex](a, b, c);
    public static void Call<T1, T2, T3, T4>(int acceleratorIndex, List<Action<T1, T2, T3, T4>> kernels, T1 a, T2 b, T3 c, T4 d) =>
        kernels[acceleratorIndex](a, b, c, d);
    public static void Call(int acceleratorIndex, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>>> kernels, ArrayView1D<double, Stride1D.Dense> view) => 
        kernels[acceleratorIndex](view.IntExtent, view);
    public static void Call<T>(
        int acceleratorIndex, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, T>> kernels,
        ArrayView1D<double, Stride1D.Dense> a, T b) => 
        kernels[acceleratorIndex](a.IntExtent, a, b);
    public static void Call<T1, T2>(int acceleratorIndex, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, T1, T2>> kernels,
        ArrayView1D<double, Stride1D.Dense> a, T1 b, T2 c) => 
        kernels[acceleratorIndex](a.IntExtent, a, b, c);
    public static void Call<T1, T2, T3>(int acceleratorIndex, List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, T1, T2, T3>> kernels,
        ArrayView1D<double, Stride1D.Dense> a, T1 b, T2 c, T3 d) => 
        kernels[acceleratorIndex](a.IntExtent, a, b, c, d);
    #endregion
    
    private static MemoryBuffer1D<double, Stride1D.Dense> Allocate(int acceleratorIndex, int size) => Accelerators[acceleratorIndex].Allocate1D<double>(size);
    private static MemoryBuffer1D<double, Stride1D.Dense> TryGetFrom(int acceleratorIndex, int size)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive", nameof(size));
        var pool = _pool[acceleratorIndex];
        MemoryBuffer1D<double, Stride1D.Dense> buffer;
        
        if (pool.TryGetValue(size, out var stack) && stack.Count > 0) buffer = stack.Pop();
        else buffer = Allocate(acceleratorIndex, size);
        
        Call(buffer.AcceleratorIndex(), FillKernels, buffer.View, 0.0);
        return buffer;
    }
    #endregion
}