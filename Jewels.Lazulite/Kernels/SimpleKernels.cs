using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite.Kernels;

public static class SimpleKernels
{
    public static void FillKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> view, float value) => view[index] = value;
    public static void ZeroKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> view) => view[index] = 0;
    
    public static void CopyKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> source, ArrayView1D<float, Stride1D.Dense> destination) => destination[index] = source[index];
    public static void ConcatKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> result)
    {
        if (index < a.Length)
            result[index] = a[index];
        else
            result[index] = b[index - a.Length];
    }

    public static void SliceKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> source,
        ArrayView1D<float, Stride1D.Dense> dest,
        int offset) =>
        dest[index] = source[index + offset];
}