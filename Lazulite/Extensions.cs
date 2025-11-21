using ILGPU;
using ILGPU.Runtime;
using Lazulite.Values;

namespace Lazulite;

public static class MemoryBufferExtensions
{
    public static int AcceleratorIndex(this MemoryBuffer1D<double, Stride1D.Dense> buffer) => Compute.GetAcceleratorIndex(buffer.Accelerator);
    public static AcceleratorStream GetStream(this MemoryBuffer1D<double, Stride1D.Dense> buffer) => Compute.GetStream(buffer.Accelerator);
    public static MemoryBuffer1D<double, Stride1D.Dense> Defer(this MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        Compute.DeferReturn(buffer);
        return buffer;
    }
    public static void Return(this MemoryBuffer1D<double, Stride1D.Dense> buffer) => Compute.Return(buffer);
}

public static class ArrayViewExtensions
{
    public static int AcceleratorIndex(this ArrayView1D<double, Stride1D.Dense> view) => Compute.GetAcceleratorIndex(view.GetAccelerator());
}

public static class AcceleratorExtensions
{
    public static int AcceleratorIndex(this Accelerator accelerator) => Compute.GetAcceleratorIndex(accelerator);
}

public static class ValueExtensions
{
    public static ScalarValue AsScalar(this Value<double> value) => new(value.Data);
    public static VectorValue AsVector(this Value<double[]> value) => new(value.Data);
    public static MatrixValue AsMatrix(this Value<double[,]> value) => new(value.Data);
    public static Value3 AsValue3(this Value<double[,,]> value) => new(value.Data);
}