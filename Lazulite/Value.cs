using ILGPU;
using ILGPU.Runtime;

namespace Lazulite;

public abstract class Value<T>(MemoryBuffer1D<double, Stride1D.Dense> data) : IDisposable
    where T : notnull
{
    public MemoryBuffer1D<double, Stride1D.Dense> Data { get; set; } = data;
    public int[] Shape => GetShape(Data.IntExtent);
    public int TotalSize => (int)Data.Length;
    public int AcceleratorIndex => Compute.GetAcceleratorIndex(Data.Accelerator);

    public T ToHost()
    {
        Compute.Synchronize(AcceleratorIndex);
        return Unroll(Data.View.GetAsArray1D());
    }
    public void FromHost(T value) => Data.CopyFromCPU(Roll(value));
    public void UpdateWith(Value<T> other) => Data.CopyFrom(other.Data);

    public void Dispose() => Compute.DeferReturn(Data);

    ~Value() => Dispose();

    public abstract T Unroll(double[] rolled);
    public abstract double[] Roll(T value);
    public abstract int[] GetShape(Index1D index);
    public abstract Value<T> Create(MemoryBuffer1D<double, Stride1D.Dense> buffer);
    
    public static implicit operator T(Value<T> value) => value.ToHost();
}