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
    public static Context Context { get; }

    private static bool _disposed;
    private static readonly ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentStack<MemoryBuffer1D<float, Stride1D.Dense>>>> _pool = []; // _pool[aidx] -> stacks[size] -> buffers
    #endregion

    static Compute()
    {
        Context = Context.Create(b => b
            .Default()
            .EnableAlgorithms()
            .AllAccelerators());

        HashSet<(AcceleratorType, string, long)> seen = [];

        int aidx = 0; // soon id like to sort this by performance
        foreach (Device device in Context.Devices.OrderByDescending(d => d.MemorySize).Where(device => seen.Add((device.AcceleratorType, device.Name, device.MemorySize))))
        {
            Accelerators[aidx] = device.CreateAccelerator(Context);
            AcceleratorIndices[Accelerators[aidx].Name] = aidx;
            
            _pool[aidx] = [];
            
            aidx++;
        }
        InitializeCuBlas();
        
        AppDomain.CurrentDomain.ProcessExit += (_, _) => Dispose();
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

    private static void Dispose()
    {
        if (_disposed) return;
        GC.WaitForPendingFinalizers();
        ClearAll();
        Context.Dispose();
        foreach (var accelerator in Accelerators.Values) accelerator.Dispose();
        foreach (var blas in _cublasHandles.Values) blas?.Dispose();
        CleanupCuBlas();
        _disposed = true;
    }
    #endregion
    #region Accelerator Management
    public static int RequestAccelerator(bool gpu = true)
    {
        Accelerator accelerator;
        
        if (gpu) accelerator = Accelerators.Values.FirstOrDefault(a => a is CudaAccelerator) ?? Accelerators.Values.First();
        else accelerator = Accelerators.Values.First();
        
        var aidx = GetAcceleratorIndex(accelerator);
        return aidx;
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
    public static void Call<T>(int aidx, KernelStorage<Action<Index1D, T>> kernel, Index1D i, T a)
        where T : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a);
    }

    public static void Call<T1, T2>(int aidx, KernelStorage<Action<Index1D, T1, T2>> kernel, Index1D i, T1 a, T2 b)
        where T1 : struct
        where T2 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b);
    }

    public static void Call<T1, T2, T3>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3>> kernel, Index1D i, T1 a, T2 b, T3 c)
        where T1 : struct
        where T2 : struct
        where T3 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c);
    }

    public static void Call<T1, T2, T3, T4>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d);
    }

    public static void Call<T1, T2, T3, T4, T5>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e);
    }

    public static void Call<T1, T2, T3, T4, T5, T6>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e,
        T6 f, T7 g)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d,
        T5 e, T6 f, T7 g, T8 h)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action); 
        kernel[aidx]!(i, a, b, c, d, e, f, g, h);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9>> kernel, Index1D i, T1 a, T2 b, T3 c,
        T4 d, T5 e, T6 f, T7 g, T8 h, T9 j) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>> kernel, Index1D i, T1 a,
        T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k) where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(int aidx, KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>> kernel,
        Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k, T11 l) where T1 : struct
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
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(int aidx,
        KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k, T11 l,
        T12 m) where T1 : struct
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
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l, m);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(int aidx,
        KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j, T10 k,
        T11 l, T12 m, T13 n) where T1 : struct
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
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l, m, n);
    }

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>(int aidx,
        KernelStorage<Action<Index1D, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>> kernel, Index1D i, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f, T7 g, T8 h, T9 j,
        T10 k, T11 l, T12 m, T13 n, T14 o) where T1 : struct
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
        where T14 : struct
    {
        kernel[aidx] ??= Accelerators[aidx].LoadAutoGroupedStreamKernel(kernel.Action);
        kernel[aidx]!(i, a, b, c, d, e, f, g, h, j, k, l, m, n, o);
    }

    public static void Call(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>>> kernel, ArrayView1D<float, Stride1D.Dense> a) 
        => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a);
    public static void Call<T>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T>> kernel, ArrayView1D<float, Stride1D.Dense> a, T b)
        where T : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b);
    public static void Call<T1, T2>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2>> kernel, ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c)
        where T1 : struct
        where T2 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c);

    public static void Call<T1, T2, T3>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3>> kernel, ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c,
        T3 d)
        where T1 : struct
        where T2 : struct
        where T3 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d);

    public static void Call<T1, T2, T3, T4>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4>> kernel, ArrayView1D<float, Stride1D.Dense> a,
        T1 b, T2 c, T3 d, T4 e)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e);

    public static void Call<T1, T2, T3, T4, T5>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f);

    public static void Call<T1, T2, T3, T4, T5, T6>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g);

    public static void Call<T1, T2, T3, T4, T5, T6, T7>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9>(KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k)
        where T1 : struct
        where T2 : struct
        where T3 : struct
        where T4 : struct
        where T5 : struct
        where T6 : struct
        where T7 : struct
        where T8 : struct
        where T9 : struct
        where T10 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l)
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
        where T11 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l, T12 m)
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
        where T12 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l, m);

    public static void Call<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>> kernel,
        ArrayView1D<float, Stride1D.Dense> a, T1 b, T2 c, T3 d, T4 e, T5 f, T6 g, T7 h, T8 i, T9 j, T10 k, T11 l, T12 m, T13 n)
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
        where T13 : struct => Call(a.AcceleratorIndex(), kernel, a.IntExtent, a, b, c, d, e, f, g, h, i, j, k, l, m, n);

    #endregion
    #region Binary Calls
    public static MemoryBuffer1D<float, Stride1D.Dense> BinaryCall(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernel,
        MemoryBuffer1D<float, Stride1D.Dense> a, 
        MemoryBuffer1D<float, Stride1D.Dense> b) => Encase(a, r => Call(kernel, r, a, b));

    public static Value<T> BinaryCall<T>(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernel, Value<T> a, Value<T> b) 
        where T : notnull => a.CreateAlike(BinaryCall(kernel, a.Data, b.Data));
    #endregion
    #region Unary Calls
    public static MemoryBuffer1D<float, Stride1D.Dense> UnaryCall(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels,
        MemoryBuffer1D<float, Stride1D.Dense> a) => Encase(a, r => Call(kernels, r, a));
    
    public static Value<T> UnaryCall<T>(
        KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> kernels, Value<T> a) 
        where T : notnull => a.CreateAlike(UnaryCall(kernels, a.Data));
    #endregion
    #region Call Chains
    public static void CallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial, 
        MemoryBuffer1D<float, Stride1D.Dense> result, 
        params Action<MemoryBuffer1D<float, Stride1D.Dense>>[] ops)
    {
        Call(CopyKernel, initial, result);
        foreach (var op in ops) op(result);
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> CallChain(
        MemoryBuffer1D<float, Stride1D.Dense> initial,
        params Action<MemoryBuffer1D<float, Stride1D.Dense>>[] ops) => Encase(initial, r => CallChain(initial, r, ops));
    #endregion
    #endregion
    
    public static MemoryBuffer1D<float, Stride1D.Dense> Allocate(int aidx, int size) => Accelerators[aidx].Allocate1D<float>(size);
    private static MemoryBuffer1D<float, Stride1D.Dense> TryGetFrom(int aidx, int size)
    {
        MemoryBuffer1D<float, Stride1D.Dense> buffer;
        
        if (_pool[aidx].TryGetValue(size, out var stack))
            buffer = stack.TryPop(out var result) ? result : Allocate(aidx, size);
        else buffer = Allocate(aidx, size);

        Call(FillKernel, buffer, 0);
        return buffer;
    }
    #endregion
}