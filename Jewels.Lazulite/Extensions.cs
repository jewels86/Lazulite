using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public static class MemoryBufferExtensions
{
    public static int AcceleratorIndex(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.Instance.GetAcceleratorIndex(buffer.Accelerator);
    public static AcceleratorStream GetStream(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.Instance.GetStream(buffer.Accelerator);
    public static void Return(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.Instance.Return(buffer);

    public static MemoryBuffer1D<float, Stride1D.Dense> DeferReturn(this MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        Compute.Instance.DeferReturn(buffer);
        return buffer;
    }
}

public static class ArrayViewExtensions
{
    public static int AcceleratorIndex(this ArrayView1D<float, Stride1D.Dense> view) => Compute.Instance.GetAcceleratorIndex(view.GetAccelerator());
}

public static class ValueExtensions
{
    public static ScalarValue AsScalar(this Value<float> value) => new(value.Data);
    public static VectorValue AsVector(this Value<float> value) => new(value.Data);
    public static VectorValue AsVector(this Value<float[]> value) => new(value.Data);
    public static MatrixValue AsMatrix(this Value<float[,]> value) => new(value.Data, value.Shape);
    public static TensorValue3 AsTensorValue3(this Value<float[,,]> value) => new(value.Data, value.Shape);

    public static Value<T> NonDisposable<T>(this Value<T> value) where T : notnull
    {
        value.Disposable = false;
        return value;
    }
}