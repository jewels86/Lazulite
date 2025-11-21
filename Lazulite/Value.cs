using ILGPU;
using ILGPU.Runtime;

namespace Lazulite;

public abstract class Value<T>(MemoryBuffer1D<double, Stride1D.Dense> data, int[] shape) : IDisposable
    where T : notnull
{
    public MemoryBuffer1D<double, Stride1D.Dense> Data { get; } = data;
    public int[] Shape { get; protected set; } = shape;
    public int TotalSize => (int)Data.Length;
    public int AcceleratorIndex => Computation.GetAcceleratorIndex(Data.Accelerator);

    public T ToHost()
    {
        Computation.Synchronize(AcceleratorIndex);
        return Unroll(Data.View.GetAsArray1D());
    }

    public void FromHost(T value) => Data.CopyFromCPU(Roll(value));

    public void Dispose() => Computation.DeferReturn(Data);

    public abstract T Unroll(double[] rolled);
    public abstract double[] Roll(T value);
}