using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Values;

public class VectorValue : Value<float[]>
{
    public VectorValue(float[] value, int aidx) : base(Compute.Get(aidx, value.Length)) => FromHost(value);
    public VectorValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer) { }
    
    public override float[] Unroll(float[] rolled) => rolled;
    public override float[] Roll(float[] value) => value;
    public override int[] GetShape(Index1D index) => [index.X];
    public override VectorValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer) => new(buffer);
    public override VectorProxy ToProxy() => new(this);

    public static VectorValue operator +(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseAddKernels, a, b).AsVector();
    public static VectorValue operator -(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseSubtractKernels, a, b).AsVector();
    public static VectorValue operator *(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseMultiplyKernels, a, b).AsVector();
    public static VectorValue operator *(VectorValue a, ScalarValue b) => new(Compute.BinaryCall(Compute.ElementwiseScalarMultiplyKernels, a, b));
    public static VectorValue operator /(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseDivideKernels, a, b).AsVector();
    public static VectorValue operator /(VectorValue a, ScalarValue b) => new(Compute.BinaryCall(Compute.ElementwiseScalarDivideKernels, a, b));
    public static VectorValue operator -(VectorValue a) => Compute.UnaryCall(Compute.ElementwiseNegateKernels, a).AsVector();
    public static VectorValue operator %(VectorValue a, VectorValue b) => Compute.BinaryCall(Compute.ElementwiseModuloKernels, a, b).AsVector();

    public ScalarValue Sum()
    {
        var result = new ScalarValue(0, AcceleratorIndex);
        Operations.Sum(this, result);
        return result;
    }

    public ScalarValue Dot(VectorValue b) => (this * b).Defer().Sum();
}

public class VectorProxy : ValueProxy<float[]>
{
    public VectorProxy(VectorValue value) : base(value) { }
    
    public float this[int i] => FlatData[i];
    public override float Get(int[] index) => FlatData[index[0]];
    public override float[] ToHost() => FlatData;
}