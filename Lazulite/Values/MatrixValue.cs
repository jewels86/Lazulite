using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Values;

public class MatrixValue : Value<double[,]>
{
    public MatrixValue(double[,] value, int aidx) 
        : base(Compute.Get(aidx, value.GetLength(0) * value.GetLength(1))) => FromHost(value);
    public MatrixValue(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer) { }
    
    public override double[] Roll(double[,] value)
    {
        var (rows, cols) = (value.GetLength(0), value.GetLength(1));
        return Roll(value, rows, cols);
    }
    public override double[,] Unroll(double[] rolled)
    {
        var (rows, cols) = (Shape[0], Shape[1]);
        return Unroll(rolled, rows, cols);
    }

    public static double[] Roll(double[,] value, int rows, int cols)
    {
        var vector = new double[rows * cols];
        for (int i = 0; i < rows; i++) 
        for (int j = 0; j < cols; j++)
            vector[IndexOf(i, j, cols)] = value[i, j];
        return vector;
    }

    public static double[,] Unroll(double[] rolled, int rows, int cols)
    {
        var matrix = new double[rows, cols];
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

    public override Value<double[,]> Create(MemoryBuffer1D<double, Stride1D.Dense> buffer) => new MatrixValue(buffer);

    public static int IndexOf(int row, int col, int cols) => row * cols + col;
    public static (int row, int col) FromIndex(int index, int cols) => (index / cols, index % cols);
    
    public static MatrixValue operator +(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseAddKernels, a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseSubtractKernels, a, b).AsMatrix();
    public static MatrixValue operator *(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseMultiplyKernels, a, b).AsMatrix();
    public static MatrixValue operator /(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseDivideKernels, a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a) => Compute.UnaryCall(Compute.ElementwiseNegateKernels, a).AsMatrix();
    public static MatrixValue operator %(MatrixValue a, MatrixValue b) => Compute.BinaryCall(Compute.ElementwiseModuloKernels, a, b).AsMatrix();
}