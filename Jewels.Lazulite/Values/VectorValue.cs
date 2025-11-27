using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public class VectorValue : Value<float[]>
{
    public VectorValue(float[] value, int aidx) : base(Compute.Instance.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public VectorValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }
    
    public override float[] Unroll(float[] rolled) => rolled;
    public override float[] Roll(float[] value) => value;
    public override VectorValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override VectorProxy ToProxy() => new(this);

    public static VectorValue operator +(VectorValue a, VectorValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseAddKernels, a, b).AsVector();
    public static VectorValue operator -(VectorValue a, VectorValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseSubtractKernels, a, b).AsVector();
    public static VectorValue operator *(VectorValue a, VectorValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseMultiplyKernels, a, b).AsVector();
    public static VectorValue operator *(VectorValue a, ScalarValue b) => new(Compute.Instance.BinaryCall(Compute.Instance.ElementwiseScalarMultiplyKernels, a, b));
    public static VectorValue operator /(VectorValue a, VectorValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseDivideKernels, a, b).AsVector();
    public static VectorValue operator /(VectorValue a, ScalarValue b) => new(Compute.Instance.BinaryCall(Compute.Instance.ElementwiseScalarDivideKernels, a, b));
    public static VectorValue operator -(VectorValue a) => Compute.Instance.UnaryCall(Compute.Instance.ElementwiseNegateKernels, a).AsVector();
    public static VectorValue operator %(VectorValue a, VectorValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseModuloKernels, a, b).AsVector();

    public ScalarValue Sum()
    {
        var result = new ScalarValue(0, AcceleratorIndex);
        Compute.Instance.Sum(this, result);
        return result;
    }

    public ScalarValue Dot(VectorValue b)
    {
        var result = new ScalarValue(0, AcceleratorIndex);
        Compute.Instance.Dot(this, b, result);
        return result;
    }
}

public class VectorProxy(VectorValue value) : ValueProxy<float[]>(value)
{
    public float this[int i] => FlatData[i];
    public override float Get(int[] index) => FlatData[index[0]];
    public override float[] ToHost() => FlatData;
}