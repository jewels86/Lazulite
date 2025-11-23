
using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Values;

public class ScalarValue : Value<float>
{
    public ScalarValue(float value, int aidx) : base(Compute.Get(aidx, 1)) => FromHost(value);
    public ScalarValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer) { }
    
    public override float Unroll(float[] rolled) => rolled[0];
    public override float[] Roll(float value) => [value];
    public override int[] GetShape(Index1D index) => [];
    public override Value<float> Create(MemoryBuffer1D<float, Stride1D.Dense> buffer) => new ScalarValue(buffer);
    
    public static ScalarValue operator +(ScalarValue a, ScalarValue b) => Compute.BinaryCall(Compute.ElementwiseAddKernels, a, b).AsScalar();
    public static ScalarValue operator -(ScalarValue a, ScalarValue b) => Compute.BinaryCall(Compute.ElementwiseSubtractKernels, a, b).AsScalar();
    public static ScalarValue operator *(ScalarValue a, ScalarValue b) => Compute.BinaryCall(Compute.ElementwiseMultiplyKernels, a, b).AsScalar();
    public static ScalarValue operator /(ScalarValue a, ScalarValue b) => Compute.BinaryCall(Compute.ElementwiseDivideKernels, a, b).AsScalar();
    public static ScalarValue operator -(ScalarValue a) => Compute.UnaryCall(Compute.ElementwiseNegateKernels, a).AsScalar();
    public static ScalarValue operator %(ScalarValue a, ScalarValue b) => Compute.BinaryCall(Compute.ElementwiseModuloKernels, a, b).AsScalar();
}