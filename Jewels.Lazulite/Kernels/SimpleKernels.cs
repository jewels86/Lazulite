using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite.Kernels;

public static class SimpleKernels
{
    public static void FillKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> view, float value) => view[index] = value;
    public static void ZeroKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> view) => view[index] = 0;
    
    public static void CopyKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> source, ArrayView1D<float, Stride1D.Dense> destination) => destination[index] = source[index];
}