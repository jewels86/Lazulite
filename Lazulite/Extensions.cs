using ILGPU;
using ILGPU.Runtime;

namespace Lazulite;

public static class MemoryBufferExtensions
{
    public static int AcceleratorIndex(this MemoryBuffer1D<double, Stride1D.Dense> buffer) => Computation.GetAcceleratorIndex(buffer.Accelerator);
    public static AcceleratorStream GetStream(this MemoryBuffer1D<double, Stride1D.Dense> buffer) => Computation.GetStream(buffer.Accelerator);
    public static MemoryBuffer1D<double, Stride1D.Dense> Defer(this MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        Computation.DeferReturn(buffer);
        return buffer;
    }
    public static void Return(this MemoryBuffer1D<double, Stride1D.Dense> buffer) => Computation.Return(buffer);
}

public static class AcceleratorExtensions
{
    public static int AcceleratorIndex(this Accelerator accelerator) => Computation.GetAcceleratorIndex(accelerator);
}