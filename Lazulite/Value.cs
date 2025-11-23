using ILGPU;
using ILGPU.Runtime;

namespace Lazulite;

public abstract class Value<T>(MemoryBuffer1D<float, Stride1D.Dense> data) : IDisposable
    where T : notnull
{
    public MemoryBuffer1D<float, Stride1D.Dense> Data { get; set; } = data;
    public int[] Shape => GetShape(Data.IntExtent);
    public int TotalSize => (int)Data.Length;
    public int AcceleratorIndex => Compute.GetAcceleratorIndex(Data.Accelerator);
    public bool IsValid { get; private set; } = true; // this may be false if the buffer was deferred but wasn't returned yet- it can still be used during that iteration

    public T ToHost()
    {
        Compute.Synchronize(AcceleratorIndex);
        return Unroll(Data.View.GetAsArray1D());
    }
    public void FromHost(T value) => Data.CopyFromCPU(Roll(value));
    public void UpdateWith(Value<T> other)
    {
        Data.CopyFrom(other.Data);
    }

    public void Dispose()
    {
        if (!IsValid) return;
        Compute.DeferReturn(Data);
        IsValid = false;
    }

    public abstract T Unroll(float[] rolled);
    public abstract float[] Roll(T value);
    public abstract int[] GetShape(Index1D index);
    public abstract Value<T> Create(MemoryBuffer1D<float, Stride1D.Dense> buffer);
    public abstract ValueProxy<T> ToProxy();
    
    public static implicit operator T(Value<T> value) => value.ToHost();
    public static implicit operator MemoryBuffer1D<float, Stride1D.Dense>(Value<T> value) => value.Data;
}

public abstract class ValueProxy<T> where T : notnull
{
    public float[] FlatData { get; }
    public int[] Shape { get; }

    protected ValueProxy(Value<T> data)
    {
        FlatData = data.Data.View.GetAsArray1D();
        Shape = data.Shape;
    }
    protected ValueProxy(float[] flatData, int[] shape)
    {
        FlatData = flatData;
        Shape = shape;
    }

    public abstract float Get(int[] index);
    public abstract T ToHost();
}