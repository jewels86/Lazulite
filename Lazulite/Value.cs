using ILGPU;
using ILGPU.Runtime;

namespace Lazulite;

public abstract class Value<T>(MemoryBuffer1D<double, Stride1D.Dense> data, int[] shape) : IDisposable
    where T : notnull
{
    public MemoryBuffer1D<double, Stride1D.Dense> Data { get; set; } = data;
    public int[] Shape { get; protected set; } = shape;
    public int TotalSize => (int)Data.Length;
    public int AcceleratorIndex => Computation.GetAcceleratorIndex(Data.Accelerator);

    public T ToHost()
    {
        Computation.Synchronize(AcceleratorIndex);
        return Unroll(Data.View.GetAsArray1D());
    }
    public void FromHost(T value) => Data.CopyFromCPU(Roll(value));
    public void UpdateWith(Value<T> other) => Data.CopyFrom(other.Data);

    public void Dispose() => Computation.DeferReturn(Data);

    public abstract T Unroll(double[] rolled);
    public abstract double[] Roll(T value);
}

public class ScalarValue : Value<double>
{
    public ScalarValue(double value, int aidx) : base(Computation.Get(aidx, 1), [1]) => FromHost(value);
    
    public override double Unroll(double[] rolled) => rolled[0];
    public override double[] Roll(double value) => [value];
}

public class VectorValue : Value<double[]>
{
    public VectorValue(double[] value, int aidx) : base(Computation.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    
    public override double[] Unroll(double[] rolled) => rolled;
    public override double[] Roll(double[] value) => value;
}

public class MatrixValue : Value<double[,]>
{
    public MatrixValue(double[,] value, int aidx) 
        : base(
            Computation.Get(aidx, value.GetLength(0) * value.GetLength(1)), 
            [value.GetLength(0), value.GetLength(1)]) => FromHost(value);
    
    public override double[] Roll(double[,] value)
    {
        var rows = value.GetLength(0);
        var vector = new double[rows * value.GetLength(1)];
        for (int i = 0; i < value.GetLength(0); i++) 
        for (int j = 0; j < value.GetLength(1); j++)
            vector[IndexOf(i, j, rows)] = value[i, j];
        return vector;
    }
    public override double[,] Unroll(double[] rolled)
    {
        var rows = rolled.Length / Shape[1];
        var matrix = new double[rows, Shape[1]];
        for (int i = 0; i < rows; i++) for (int j = 0; j < Shape[1]; j++)
            matrix[i, j] = rolled[IndexOf(i, j, rows)];
        return matrix;
    }
    
    
    private static int IndexOf(int row, int col, int rows) => row * rows + col;
    private static (int row, int col) FromIndex(int index, int rows) => (index / rows, index % rows);
}

public class Value3 : Value<double[,,]>
{
    public Value3(double[,,] value, int aidx) 
        : base(Computation.Get(
            aidx, 
            value.GetLength(0) * value.GetLength(1) * value.GetLength(2)), 
            [value.GetLength(0), value.GetLength(1), value.GetLength(2)]) => 
        FromHost(value);
    
    public override double[] Roll(double[,,] value)
    {
        var w = value.GetLength(2);
        var vector = new double[w * w * w];
        for (int i = 0; i < w; i++) 
        for (int j = 0; j < w; j++) 
        for (int k = 0; k < w; k++)
            vector[IndexOf(i, j, k, w)] = value[i, j, k];
        return vector;
    }
    public override double[,,] Unroll(double[] rolled)
    {
        var w = rolled.Length / (Shape[0] * Shape[1]);
        var matrix = new double[Shape[0], Shape[1], w];
        for (int i = 0; i < Shape[0]; i++) 
        for (int j = 0; j < Shape[1]; j++) 
        for (int k = 0; k < w; k++)
            matrix[i, j, k] = rolled[IndexOf(i, j, k, w)];
        return matrix;
    }
    
    private static int IndexOf(int x, int y, int z, int w) => x * w * w + y * w + z;
    private static (int x, int y, int z) FromIndex(int index, int w) => (index / (w * w), (index / w) % w, index % w);
}