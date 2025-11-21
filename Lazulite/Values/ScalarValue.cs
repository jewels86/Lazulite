
using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Values;

public class ScalarValue : Value<double>
{
    public ScalarValue(double value, int aidx) : base(Compute.Get(aidx, 1)) => FromHost(value);
    public ScalarValue(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer) { }
    
    public override double Unroll(double[] rolled) => rolled[0];
    public override double[] Roll(double value) => [value];
    public override int[] GetShape(Index1D index) => [];
    public override Value<double> Create(MemoryBuffer1D<double, Stride1D.Dense> buffer) => new ScalarValue(buffer);
}

public class Value0(double value, int aidx) : ScalarValue(value, aidx);