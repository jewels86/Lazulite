using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Values;

public class VectorValue : Value<double[]>
{
    public VectorValue(double[] value, int aidx) : base(Compute.Get(aidx, value.Length)) => FromHost(value);
    public VectorValue(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer) { }
    
    public override double[] Unroll(double[] rolled) => rolled;
    public override double[] Roll(double[] value) => value;
    public override int[] GetShape(Index1D index) => [index.X];
    public override Value<double[]> Create(MemoryBuffer1D<double, Stride1D.Dense> buffer) => new VectorValue(buffer);
}

public class Value1(double[] value, int aidx) : VectorValue(value, aidx);