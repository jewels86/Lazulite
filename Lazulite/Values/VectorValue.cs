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
    
    public static VectorValue operator +(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseAddKernels, a, b).AsVector();
    public static VectorValue operator -(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseSubtractKernels, a, b).AsVector();
    public static VectorValue operator *(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseMultiplyKernels, a, b).AsVector();
    public static VectorValue operator /(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseDivideKernels, a, b).AsVector();
    public static VectorValue operator -(VectorValue a) => Compute.UnaryCall(Compute.ElementwiseNegateKernels, a).AsVector();
    public static VectorValue operator %(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseModuloKernels, a, b).AsVector();

    public ScalarValue Sum()
    {
        var result = new ScalarValue(0, AcceleratorIndex);
        Operations.Sum(this, result);
        return result;
    }

    public ScalarValue Dot(VectorValue b)
    {
        return (this * b).Defer().Sum();
    }
}