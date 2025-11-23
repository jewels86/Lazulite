using ILGPU.Runtime.Cuda;
using Lazulite;

namespace Testing;

class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine(Compute.Accelerators[2] is CudaAccelerator cudaAccelerator ? cudaAccelerator.DriverVersion.ToString() : ":(");
        //Compute.EnsureWarmup();
        
        //SimpleTests.FillTest(false);
        //SimpleTests.FillTest(true);
        
        //SimpleTests.SimpleMathTest(false);
        //SimpleTests.SimpleMathTest(true);
        
        SimpleTests.ScalarTest(true);
        //SimpleTests.PhysicsTest(true);
        //SimpleTests.ParallelProcessingTest(true);
        //ValueTests.MathTest(true);
        
        Compute.ClearAll();
    }
}