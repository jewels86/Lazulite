using ILGPU;
using ILGPU.Runtime;
using Lazulite.Kernels;
using static Lazulite.Kernels.SimpleKernels;

namespace Lazulite;

public static partial class Computation
{
    #region Simple Kernels
    public static Action<AcceleratorStream, Index1D, ArrayView1D<double, Stride1D.Dense>, double> FillKernel;
    public static Action<AcceleratorStream, Index1D, ArrayView1D<double, Stride1D.Dense>> ZeroKernel;
    public static Action<AcceleratorStream, Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>> CopyKernel;
    #endregion

    public static void InitializeKernels()
    {
        #region Simple Kernels
        FillKernel = Accelerators[0].LoadAutoGroupedKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, double>(SimpleKernels.FillKernel);
        ZeroKernel = Accelerators[0].LoadAutoGroupedKernel<Index1D, ArrayView1D<double, Stride1D.Dense>>(SimpleKernels.ZeroKernel);
        CopyKernel = Accelerators[0].LoadAutoGroupedKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(SimpleKernels.CopyKernel);
        #endregion
    }
}