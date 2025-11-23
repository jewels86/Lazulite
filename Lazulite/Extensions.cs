using ILGPU;
using ILGPU.Runtime;
using Lazulite.Values;

namespace Lazulite;

public static class DisposableExtensions
{
    public static T Defer<T>(this T disposable) where T : IDisposable
    {
        disposable.Dispose();
        return disposable;
    }
}

public static class MemoryBufferExtensions
{
    public static int AcceleratorIndex(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.GetAcceleratorIndex(buffer.Accelerator);
    public static AcceleratorStream GetStream(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.GetStream(buffer.Accelerator);
    public static void Return(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.Return(buffer);

    public static MemoryBuffer1D<float, Stride1D.Dense> DeferReturn(this MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        Compute.DeferReturn(buffer);
        return buffer;
    }
}

public static class ArrayViewExtensions
{
    public static int AcceleratorIndex(this ArrayView1D<float, Stride1D.Dense> view) => Compute.GetAcceleratorIndex(view.GetAccelerator());
}

public static class AcceleratorExtensions
{
    public static int AcceleratorIndex(this Accelerator accelerator) => Compute.GetAcceleratorIndex(accelerator);
}

public static class ValueExtensions
{
    public static ScalarValue AsScalar(this Value<float> value) => new(value.Data);
    public static VectorValue AsVector(this Value<float[]> value) => new(value.Data);
    public static MatrixValue AsMatrix(this Value<float[,]> value) => new(value.Data, value.Shape);
    public static TensorValue3 AsTensorValue3(this Value<float[,,]> value) => new(value.Data, value.Shape);
}