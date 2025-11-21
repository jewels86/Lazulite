using ILGPU;
using ILGPU.Runtime;

namespace Lazulite.Kernels;

public static class SimpleKernels
{
    public static void FillKernel(Index1D index, ArrayView1D<double, Stride1D.Dense> view, double value) => view[index] = value;
    public static void ZeroKernel(Index1D index, ArrayView1D<double, Stride1D.Dense> view) => view[index] = 0;
    
    public static void CopyKernel(Index1D index, ArrayView1D<double, Stride1D.Dense> source, ArrayView1D<double, Stride1D.Dense> destination) => destination[index] = source[index];
}