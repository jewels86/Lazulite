using ILGPU;
using ILGPU.Runtime;
using Lazulite.Kernels;
using static Lazulite.Kernels.SimpleKernels;
using static Lazulite.Kernels.ElementwiseKernels;

namespace Lazulite;

public static partial class Compute
{
    #region Simple Kernels
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, double>> FillKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>>> ZeroKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> CopyKernels = [];
    #endregion
    #region Elementwise Kernels
    #region Binary
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseAddKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseSubtractKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseMultiplyKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseDivideKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwisePowerKernels = [];
    #endregion
    #region Unary
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseExpKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseLogKernels = [];
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>> ElementwiseSqrtKernels = [];
    #endregion
    public readonly static List<Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>,
        ArrayView1D<double, Stride1D.Dense>>> ElementwiseScalarPowerKernels = [];
    #endregion

    public static void InitializeKernels()
    {
        foreach (var accelerator in Accelerators)
        {
            #region Simple Kernels
            FillKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, double>(FillKernel));
            ZeroKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>>(ZeroKernel));
            CopyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(CopyKernel));
            #endregion
            #region Elementwise Kernels
            #region Binary
            ElementwiseAddKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseAddKernel));
            ElementwiseSubtractKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseSubtractKernel));
            ElementwiseMultiplyKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseMultiplyKernel));
            ElementwiseDivideKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseDivideKernel));
            ElementwisePowerKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwisePowerKernel));
            #endregion
            #region Unary
            ElementwiseExpKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseExpKernel));
            ElementwiseLogKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseLogKernel));
            ElementwiseSqrtKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(ElementwiseSqrtKernel));
            #endregion
            ElementwiseScalarPowerKernels.Add(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
                ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(ElementwiseScalarPowerKernel));
            #endregion
        }
    }
}