using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public class MatrixValue : Value<float[,]>
{
    public MatrixValue(float[,] value, int aidx) : base(
        Compute.Instance.Get(aidx, value.GetLength(0) * value.GetLength(1)),
        [value.GetLength(0), value.GetLength(1)]) => FromHost(value);
    public MatrixValue(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : base(buffer, shape) { }
    
    public override float[] Roll(float[,] value) => MatrixProxy.Roll(value);
    public override float[,] Unroll(float[] rolled) => MatrixProxy.Unroll(rolled, Shape[1]);

    public override MatrixValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, shape);
    public override MatrixProxy ToProxy() => new(this);
    
    public static MatrixValue operator +(MatrixValue a, MatrixValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseAddKernels, a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a, MatrixValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseSubtractKernels, a, b).AsMatrix();
    public static MatrixValue operator *(MatrixValue a, MatrixValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseMultiplyKernels, a, b).AsMatrix();
    public static MatrixValue operator /(MatrixValue a, MatrixValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseDivideKernels, a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a) => Compute.Instance.UnaryCall(Compute.Instance.ElementwiseNegateKernels, a).AsMatrix();
    public static MatrixValue operator %(MatrixValue a, MatrixValue b) => Compute.Instance.BinaryCall(Compute.Instance.ElementwiseModuloKernels, a, b).AsMatrix();

    public MatrixValue MatrixMultiply(MatrixValue other)
    {
        // this should be m * k, they should be k * n
        if (Shape[1] != other.Shape[0]) throw new InvalidOperationException("Matrix dimensions are not compatible.");
        var result = Compute.Instance.Get(AcceleratorIndex, Shape[0] * other.Shape[1]);
        Compute.Instance.MatrixMultiply(this, other, result, Shape[0], Shape[1], other.Shape[1]);
        return new(result, [Shape[0], other.Shape[1]]);
    }
}

public class MatrixProxy(Value<float[,]> value) : ValueProxy<float[,]>(value)
{
    public float this[int i, int j] => FlatData[IndexOf(i, j, Shape[1])];
    public override float Get(int[] index) => this[index[0], index[1]];
    public override float[,] ToHost() => Unroll(FlatData, Shape[1]);
    
    public static float[] Roll(float[,] value)
    {
        var (rows, cols) = (value.GetLength(0), value.GetLength(1));
        var vector = new float[rows * cols];
        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                vector[IndexOf(i, j, cols)] = value[i, j];
        });
        return vector;
    }
    
    public static float[,] Unroll(float[] rolled, int cols)
    {
        var rows = rolled.Length / cols;
        var matrix = new float[rows, cols];
        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                matrix[i, j] = rolled[IndexOf(i, j, cols)];
        });
        return matrix;
    }
    
    public static int IndexOf(int row, int col, int cols) => row * cols + col;
    public static (int row, int col) FromIndex(int index, int cols) => (index / cols, index % cols);
}