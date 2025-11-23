using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Values;

public class MatrixValue : Value<float[,]>
{
    public MatrixValue(float[,] value, int aidx) 
        : base(Compute.Get(aidx, value.GetLength(0) * value.GetLength(1))) => FromHost(value);
    public MatrixValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer) { }
    
    public override float[] Roll(float[,] value)
    {
        var (rows, cols) = (value.GetLength(0), value.GetLength(1));
        return Roll(value, rows, cols);
    }
    public override float[,] Unroll(float[] rolled)
    {
        var (rows, cols) = (Shape[0], Shape[1]);
        return Unroll(rolled, rows, cols);
    }

    public static float[] Roll(float[,] value, int rows, int cols)
    {
        var vector = new float[rows * cols];
        for (int i = 0; i < rows; i++) 
        for (int j = 0; j < cols; j++)
            vector[IndexOf(i, j, cols)] = value[i, j];
        return vector;
    }

    public static float[,] Unroll(float[] rolled, int rows, int cols)
    {
        var matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++) 
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rolled[IndexOf(i, j, cols)];
        return matrix;
    }

    public override int[] GetShape(Index1D index)
    {
        var(row, col) = FromIndex(index.X, Shape[1]);
        return [row, col];
    }

    public override Value<float[,]> Create(MemoryBuffer1D<float, Stride1D.Dense> buffer) => new MatrixValue(buffer);

    public static int IndexOf(int row, int col, int cols) => row * cols + col;
    public static (int row, int col) FromIndex(int index, int cols) => (index / cols, index % cols);
    
    public static MatrixValue operator +(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseAddKernels, a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseSubtractKernels, a, b).AsMatrix();
    public static MatrixValue operator *(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseMultiplyKernels, a, b).AsMatrix();
    public static MatrixValue operator /(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseDivideKernels, a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a) => Compute.UnaryCall(Compute.ElementwiseNegateKernels, a).AsMatrix();
    public static MatrixValue operator %(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseModuloKernels, a, b).AsMatrix();
}