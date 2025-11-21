using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;

namespace Lazulite;

public static partial class Operations
{
    public static void Sum(MemoryBuffer1D<double, Stride1D.Dense> a, MemoryBuffer1D<double, Stride1D.Dense> result)
    {
        var aidx = a.AcceleratorIndex();
        Compute.Accelerators[aidx].Reduce<double, AddDouble>(
            Compute.GetStream(aidx),
            a.View, result.View);
    }
}