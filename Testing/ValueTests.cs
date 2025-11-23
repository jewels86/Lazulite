using System.Diagnostics;
using Lazulite;
using Lazulite.Values;

namespace Testing;

public static class ValueTests
{
    private static int _aidx = -1;
    
    public static void MathTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? $"GPU accelerator {_aidx}" : $"CPU accelerator {_aidx}");
        Stopwatch sw = new();

        using MatrixValue a = new(SimpleTests.RandomMatrix(10000, 10000), _aidx);
        using MatrixValue b = new(SimpleTests.RandomMatrix(10000, 10000), _aidx);
        using var c = a + b;
        
        sw.Start();
        float[,] matrix = c.ToHost();
        sw.Stop();
        Console.WriteLine($"ToHost elapsed time: {sw.ElapsedMilliseconds}");
        Console.WriteLine($"Matrix sum at (0, 0): {matrix[0, 0]}");

        using var d = a + b;
        sw.Restart();
        MatrixProxy proxy = d.ToProxy();
        sw.Stop();
        Console.WriteLine($"ToProxy elapsed time: {sw.ElapsedMilliseconds}");
        Console.WriteLine($"Matrix sum at (0, 0): {proxy[0, 0]}");
        
        Compute.ReleaseAccelerator(_aidx);
    }
}