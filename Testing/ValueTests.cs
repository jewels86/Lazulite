using System.Data.Common;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using Lazulite;

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

    public static void OpsTest(bool gpu)
    {
        int aidx = Compute.RequestAccelerator(gpu);
        
        var a = Compute.Get(aidx, 4);
        var b = Compute.Get(aidx, 4);
        float[] realA = [1, 2, 3, 4];
        float[] realB = [3, 4, 5, 6];
        a.CopyFromCPU(realA);
        b.CopyFromCPU(realB);
        
        using var addBuffer = Compute.Get(aidx, 4);
        using var subBuffer = Compute.Get(aidx, 4);
        using var mulBuffer = Compute.Get(aidx, 4);
        using var divBuffer = Compute.Get(aidx, 4);
        using var modBuffer = Compute.Get(aidx, 4);
        using var powBuffer = Compute.Get(aidx, 4);
        using var maxBuffer = Compute.Get(aidx, 4);
        
        Compute.Call(aidx, Compute.ElementwiseAddKernels, a.View, b.View, addBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseSubtractKernels, a.View, b.View, subBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseMultiplyKernels, a.View, b.View, mulBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseDivideKernels, a.View, b, divBuffer);
        Compute.Call(aidx, Compute.ElementwiseModuloKernels, a.View, b, modBuffer);
        Compute.Call(aidx, Compute.ElementwisePowerKernels, a.View, b, powBuffer);
        Compute.Call(aidx, Compute.ElementwiseMaxKernels, a.View, b, maxBuffer);
        
        Compute.Synchronize(aidx);
        
        Console.WriteLine($"a + b: {string.Join(',', addBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x + realB[i]))}");
        Console.WriteLine($"a - b: {string.Join(',', subBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x - realB[i]))}");
        Console.WriteLine($"a * b: {string.Join(',', mulBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x * realB[i]))}");
        Console.WriteLine($"a / b: {string.Join(',', divBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x / realB[i]))}");
        Console.WriteLine($"a % b: {string.Join(',', modBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x % realB[i]))}");
        Console.WriteLine($"a ^ b: {string.Join(',', powBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => (float)Math.Pow(x, realB[i])))}");
        Console.WriteLine($"max(a, b): {string.Join(',', maxBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => (float)Math.Max(x, realB[i])))}");

        using var expBuffer = Compute.Get(aidx, 4);
        using var logBuffer = Compute.Get(aidx, 4);
        using var sqrtBuffer = Compute.Get(aidx, 4);
        using var absBuffer = Compute.Get(aidx, 4);
        using var negateBuffer = Compute.Get(aidx, 4);
        using var tanhBuffer = Compute.Get(aidx, 4);
        using var sech2Buffer = Compute.Get(aidx, 4);
        using var naturalLogBuffer = Compute.Get(aidx, 4);
        
        Compute.Call(aidx, Compute.ElementwiseExpKernels, a.View, expBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseLogKernels, a.View, logBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseSqrtKernels, a.View, sqrtBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseAbsKernels, a.View, absBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseNegateKernels, a.View, negateBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseTanhKernels, a.View, tanhBuffer.View);
        Compute.Call(aidx, Compute.ElementwiseSech2Kernels, a.View, sech2Buffer.View);
        Compute.Call(aidx, Compute.ElementwiseNaturalLogKernels, a.View, naturalLogBuffer.View);
        
        Compute.Synchronize(aidx);
        Console.WriteLine($"exp(a): {string.Join(',', expBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Exp(x)))}");
        Console.WriteLine($"log(a): {string.Join(',', logBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Log(x)))}");
        Console.WriteLine($"sqrt(a): {string.Join(',', sqrtBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Sqrt(x)))}");
        Console.WriteLine($"abs(a): {string.Join(',', absBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(Math.Abs))}");
        Console.WriteLine($"-a: {string.Join(',', negateBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => -x))}");
        Console.WriteLine($"tanh(a): {string.Join(',', tanhBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Tanh(x)))}");
        Console.WriteLine($"sech^2(a): {string.Join(',', sech2Buffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x =>  1 / ((float)Math.Cosh(x) * Math.Cosh(x))))}");
        Console.WriteLine($"ln(a): {string.Join(',', naturalLogBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Log(x)))}");
        
        Compute.ReleaseAccelerator(aidx);
    }
}