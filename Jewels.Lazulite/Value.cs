using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public interface IValue : IDisposable 
{
    // when preforming elementwise operations, you don't need the host type, just the buffer
    // IValue lets you do that without needing to know the host type in all cases
    public MemoryBuffer1D<float, Stride1D.Dense> Data { get; }
    public int[] Shape { get; }
    public int TotalSize { get; }
    public int AcceleratorIndex { get; }
    public bool IsValid { get; }
    
    // these are used for creating new values from existing ones without knowing the host type
    // you can safely cast these to Value<T>s- Value<T> is the only class that implements IValue and it will always return the correct type
    // if for some reason they can't be casted, the object is not a Value<T>
    public IValue Zeros();
    public IValue CreateAlike(MemoryBuffer1D<float, Stride1D.Dense> buffer);
    public IValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape);
    public void UpdateWith(IValue other);
}

public abstract class Value<T>(MemoryBuffer1D<float, Stride1D.Dense> data, int[] shape) : IValue
    where T : notnull
{
    private Compute _compute => Compute.Instance;
    
    public MemoryBuffer1D<float, Stride1D.Dense> Data { get; set; } = data;
    public int[] Shape { get; } = shape;
    public int TotalSize => (int)Data.Length;
    public int AcceleratorIndex => _compute.GetAcceleratorIndex(Data.Accelerator);
    public bool IsValid { get; private set; } = true; // this may be false if the buffer was deferred but wasn't returned yet- it can still be used during that iteration
    
    public T ToHost()
    {
        _compute.Synchronize(AcceleratorIndex);
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
        _compute.DeferReturn(Data);
        IsValid = false;
    }

    public Value<T> Zeros() => Create(_compute.GetLike(this), Shape);
    public Value<T> CreateAlike(MemoryBuffer1D<float, Stride1D.Dense> buffer) => Create(buffer, Shape);

    public abstract T Unroll(float[] rolled);
    public abstract float[] Roll(T value);
    public abstract Value<T> Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape);
    public abstract ValueProxy<T> ToProxy();
    
    public static implicit operator T(Value<T> value) => value.ToHost();
    public static implicit operator MemoryBuffer1D<float, Stride1D.Dense>(Value<T> value) => value.Data;
    public static implicit operator ValueProxy<T>(Value<T> value) => value.ToProxy();
    public static implicit operator ArrayView1D<float, Stride1D.Dense>(Value<T> value) => value.Data.View;
    
    IValue IValue.Zeros() => Zeros();
    IValue IValue.CreateAlike(MemoryBuffer1D<float, Stride1D.Dense> buffer) => CreateAlike(buffer);
    IValue IValue.Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => Create(buffer, shape);
    void IValue.UpdateWith(IValue other) => UpdateWith((Value<T>)other);
}

public interface IValueProxy
{
    // these are used for the same purpose as IValue- they let you get a proxy without knowing the host type
    public float[] FlatData { get; }
    public int[] Shape { get; }
    
    // these should not be used unless you are confident in the host type
    public float Get(int[] index);
    public object ToHost();
}

public abstract class ValueProxy<T>(float[] flatData, int[] shape) : IValueProxy
    where T : notnull
{
    public float[] FlatData { get; } = flatData;
    public int[] Shape { get; } = shape;

    protected ValueProxy(Value<T> data) : this(data.Data.View.GetAsArray1D(), data.Shape) { }

    public abstract float Get(int[] index);
    public abstract T ToHost();
    
    object IValueProxy.ToHost() => ToHost();
}