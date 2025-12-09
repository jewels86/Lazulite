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

    public static VectorValue operator +(VectorValue a, VectorValue b) => Compute.Instance.Add(a, b).AsVector();
    public static VectorValue operator -(VectorValue a, VectorValue b) => Compute.Instance.Subtract(a, b).AsVector();
    public static VectorValue operator *(VectorValue a, VectorValue b) => Compute.Instance.ElementwiseMultiply(a, b).AsVector();
    public static VectorValue operator *(VectorValue a, ScalarValue b) => Compute.Instance.ScalarMultiply(a, b).AsVector();
    public static VectorValue operator /(VectorValue a, VectorValue b) => Compute.Instance.Divide(a, b).AsVector();
    public static VectorValue operator /(VectorValue a, ScalarValue b) => Compute.Instance.ScalarDivide(a, b).AsVector();
    public static VectorValue operator -(VectorValue a) => Compute.Instance.Negate(a).AsVector();

    public ScalarValue Sum() => Compute.Instance.Sum(this).AsScalar();
    public ScalarValue Dot(VectorValue b) => Compute.Instance.Dot(this, b).AsScalar();
}

public class VectorProxy(VectorValue value) : ValueProxy<float[]>(value)
{
    public float this[int i] => FlatData[i];
    public override float Get(int[] index) => FlatData[index[0]];
    public override float[] ToHost() => FlatData;
}