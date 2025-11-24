using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public abstract class Value<T>(MemoryBuffer1D<float, Stride1D.Dense> data, int[] shape) : IDisposable
    where T : notnull
{
    public MemoryBuffer1D<float, Stride1D.Dense> Data { get; set; } = data;
    public int[] Shape { get; } = shape;
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
    public abstract Value<T> Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape);
    public abstract ValueProxy<T> ToProxy();
    
    public static implicit operator T(Value<T> value) => value.ToHost();
    public static implicit operator MemoryBuffer1D<float, Stride1D.Dense>(Value<T> value) => value.Data;
}

public abstract class ValueProxy<T>(float[] flatData, int[] shape)
    where T : notnull
{
    public float[] FlatData { get; } = flatData;
    public int[] Shape { get; } = shape;

    protected ValueProxy(Value<T> data) : this(data.Data.View.GetAsArray1D(), data.Shape) { }

    public abstract float Get(int[] index);
    public abstract T ToHost();
}