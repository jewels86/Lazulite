using System.Diagnostics;

using Lazulite;

namespace Testing;

public static class SimpleTests
{
    private static int _aidx = -1;

    public static void Test(bool gpu)
    {
        _aidx = Computation.RequestAccelerator(gpu);
        Console.WriteLine(Computation.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = Stopwatch.StartNew();

        sw.Start();
        for (int i = 0; i < 20000; i++)
        {
            var buffer = Computation.GetTemp(_aidx, 5000);
            Computation.Call(_aidx, Computation.FillKernels, buffer.View, 1);
        }
        sw.Stop();
        Computation.Synchronize(_aidx);
        Console.WriteLine(sw.ElapsedMilliseconds);

        sw.Restart();
        for (int i = 0; i < 20000; i++)
        {
            var buffer = Computation.GetTemp(_aidx, 5000);
            Computation.Call(_aidx, Computation.FillKernels, buffer.View, 1);
        }
        sw.Stop();
        Console.WriteLine(sw.ElapsedMilliseconds);
        Computation.Synchronize(_aidx);
        Computation.ReleaseAccelerator(_aidx);
    }
}