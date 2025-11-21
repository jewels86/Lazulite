using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Values;

public class Value3 : Value<double[,,]>
{
    public Value3(double[,,] value, int aidx) 
        : base(Compute.Get(
                aidx, 
                value.GetLength(0) * value.GetLength(1) * value.GetLength(2))) => 
        FromHost(value);
    public Value3(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer) { }
    
    public override double[,,] Unroll(double[] rolled)
    {
        var d0 = Shape[0];
        var d1 = Shape[1];
        var d2 = Shape[2];
        var matrix = new double[d0, d1, d2];
        for (int i = 0; i < d0; i++) 
        for (int j = 0; j < d1; j++) 
        for (int k = 0; k < d2; k++)
            matrix[i, j, k] = rolled[IndexOf(i, j, k, d1, d2)];
        return matrix;
    }
    public override double[] Roll(double[,,] value)
    {
        var d0 = Shape[0];
        var d1 = Shape[1];
        var d2 = Shape[2];
        var vector = new double[d0 * d1 * d2];
        for (int i = 0; i < d0; i++) 
        for (int j = 0; j < d1; j++) 
        for (int k = 0; k < d2; k++)
            vector[IndexOf(i, j, k, d1, d2)] = value[i, j, k];
        return vector;
    }

    public override int[] GetShape(Index1D index)
    {
        var (x, y, z) = FromIndex(index.X, Shape[1], Shape[2]);
        return [x, y, z];
    }
    public override Value<double[,,]> Create(MemoryBuffer1D<double, Stride1D.Dense> buffer) => new Value3(buffer);

    public static int IndexOf(int x, int y, int z, int d1, int d2) 
        => x * (d1 * d2) + y * d2 + z;
    public static (int x, int y, int z) FromIndex(int index, int d1, int d2) 
        => (index / (d1 * d2), (index / d2) % d1, index % d2);
}