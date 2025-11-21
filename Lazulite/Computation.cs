using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Lazulite.Kernels;

namespace Lazulite;

public static partial class Computation
{
    public static List<Accelerator> Accelerators { get; } = [];
    public static List<int> Users { get; } = [];
    public static Context Context { get; private set; }
    public static bool AllowGpu { get; set; }
    public static bool GpuInUse { get; private set; }

    private static readonly List<MemoryBuffer1D<double, Stride1D.Dense>> _deferred = [];
    private static readonly List<Dictionary<int, Stack<MemoryBuffer1D<double, Stride1D.Dense>>>> _pool = []; // acceleratorIndex -> stack of buffers sorted by size

    static Computation()
    {
        Context = Context.CreateDefault();
        RefreshDevices();
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
            GpuInUse = true;
        }
    }
    public static void FlushAll()
    {
        
    }
    #region Synchronization
    public static void Synchronize(int acceleratorIndex) => Accelerators[acceleratorIndex].Synchronize();
    public static void SynchronizeAll()
    {
        foreach (Accelerator accelerator in Accelerators) accelerator.Synchronize();
    }
    #endregion
    #region Accelerator Management
    public static int RequestAccelerator()
    {
        int lowestUsers = Users.IndexOf(Users.Min());
        Users[lowestUsers]++;
        return lowestUsers;
    }

    public static void ReleaseAccelerator(int acceleratorIndex) => Users[acceleratorIndex]--;
    #endregion
    #endregion

    public static void Return(MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        if (!GpuInUse) buffer.Dispose();
        int size = (int)buffer.Length;
        ZeroKernel(GetStream(GetAcceleratorIndex(buffer.Accelerator)), size, buffer.View);
    }
    public static void DeferReturn(MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        
    }
    
    #region Helpers
    public static int GetAcceleratorIndex(Accelerator accelerator) => Accelerators.IndexOf(accelerator);
    public static AcceleratorStream GetStream(int acceleratorIndex) => Accelerators[acceleratorIndex].DefaultStream;
    public static void Call(Action) => 
    
    private static MemoryBuffer1D<double, Stride1D.Dense> Allocate(int acceleratorIndex, int size) => Accelerators[acceleratorIndex].Allocate1D<double>(size);

    private static MemoryBuffer1D<double, Stride1D.Dense> TryGetFrom(int acceleratorIndex, int size)
    {
        var pool = _pool[acceleratorIndex];
        var buffer = pool.TryGetValue(size, out var stack) ? stack.Pop() : Allocate(acceleratorIndex, size);
        
    }
    #endregion
}