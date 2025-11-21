using ILGPU;
using ILGPU.Runtime;
using Lazulite.Kernels;
using static Lazulite.Kernels.SimpleKernels;

namespace Lazulite;

public static partial class Computation
{
    #region Simple Kernels
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, double>> FillKernels = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>>> ZeroKernels = [];
    public static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> CopyKernels = [];
    #endregion

    public static void InitializeKernels()
    {
        foreach (var accelerator in Accelerators)
        {
            #region Simple Kernels
            FillKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, double>(SimpleKernels.FillKernel));
            ZeroKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>>(SimpleKernels.ZeroKernel));
            CopyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(SimpleKernels.CopyKernel));
            #endregion
        }
    }
}