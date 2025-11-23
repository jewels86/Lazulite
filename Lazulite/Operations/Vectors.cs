using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Lazulite;

public static partial class Operations
{
    public static void Sum(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> result)
    {
        var aidx = a.AcceleratorIndex();
        Compute.Accelerators[aidx].Reduce<float, AddFloat>(
            Compute.GetStream(aidx),
            a.View, result.View);
    }
}