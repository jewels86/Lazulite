using Jewels.Lazulite;

namespace Testing;

class Program
{
    public static void Main(string[] args)
    {
        using var compute = Compute.Instance;
        compute.InitializeKernelsAsync();
        compute.WaitForInitializationAsync();
        
        //SimpleTests.FillTest(false);
        SimpleTests.FillTest(true);
        
        //SimpleTests.SimpleMathTest(false);
        SimpleTests.SimpleMathTest(true);
        
        //SimpleTests.ScalarTest(true);
        SimpleTests.PhysicsTest(true);
        //SimpleTests.ParallelProcessingTest(true);
        //SimpleTests.ParallelProcessingTest(true, false);
        //SimpleTests.BigMatMulTest(true);
        SimpleTests.PoolTest(true);
        //ValueTests.MathTest(true);
        //ValueTests.OpsTest(true);
        //ValueTests.MemoryTest(true);
        
        Benchmarks.SquareBenchmark1(true);
        Benchmarks.SquareBenchmark2(true);
        Benchmarks.MatMulBenchmark(true);
        Benchmarks.MatMulBenchmark(true, 10000, 10000, 10000, 1);
    }
}
