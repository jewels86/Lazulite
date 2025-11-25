using ILGPU.Runtime.Cuda;
using Jewels.Lazulite;

namespace Testing;

class Program
{
    public static void Main(string[] args)
    {
        using var scope = new ComputeScope();
        Compute.InitializeExtraKernels();
        
        //SimpleTests.FillTest(false);
        //SimpleTests.FillTest(true);
        
        //SimpleTests.SimpleMathTest(false);
        //SimpleTests.SimpleMathTest(true);
        
        //SimpleTests.ScalarTest(true);
        //SimpleTests.PhysicsTest(true);
        //SimpleTests.ParallelProcessingTest(true, true);
        //SimpleTests.ParallelProcessingTest(true, false);
        //SimpleTests.BigMatMulTest(true);
        //ValueTests.MathTest(true);
        ValueTests.OpsTest(true);
        ValueTests.MemoryTest(true);
        
        Compute.ClearAll();
    }
}
