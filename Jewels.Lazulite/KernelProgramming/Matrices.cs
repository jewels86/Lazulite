using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public partial class KernelProgramming
{
    public static int IndexOf(int row, int col, int cols) => row * cols + col;
    public static (int row, int col) FromIndex(int index, int cols) => (index / cols, index % cols);
    
    public static float Get(ArrayView1D<float, Stride1D.Dense> array, int row, int col, int cols) => array[IndexOf(row, col, cols)];
    public static void Set(ArrayView1D<float, Stride1D.Dense> array, int row, int col, int cols, float value) => array[IndexOf(row, col, cols)] = value;
}