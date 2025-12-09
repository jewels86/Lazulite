using System.Diagnostics;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Testing;

public static class ValueTests
{
    private static Compute _compute => Compute.Instance;
    private static int _aidx = -1;
    
    public static void MathTest(bool gpu)
    {
        _aidx = _compute.RequestAccelerator(gpu);
        Console.WriteLine(_compute.IsGpuAccelerator(_aidx) ? $"GPU accelerator {_aidx}" : $"CPU accelerator {_aidx}");
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
        
        _compute.ReleaseAccelerator(_aidx);
    }

    public static void OpsTest(bool gpu)
    {
        int aidx = _compute.RequestAccelerator(gpu);
        
        var a = _compute.Get(aidx, 4);
        var b = _compute.Get(aidx, 4);
        float[] realA = [1, 2, 3, 4];
        float[] realB = [3, 4, 5, 6];
        a.CopyFromCPU(realA);
        b.CopyFromCPU(realB);
        
        using var addBuffer = _compute.Get(aidx, 4);
        using var subBuffer = _compute.Get(aidx, 4);
        using var mulBuffer = _compute.Get(aidx, 4);
        using var divBuffer = _compute.Get(aidx, 4);
        using var maxBuffer = _compute.Get(aidx, 4);
        
        // _compute.Call(_compute.ElementwiseAddKernels, a.View, b.View, addBuffer.View);
        // _compute.Call(_compute.ElementwiseSubtractKernels, a.View, b.View, subBuffer.View);
        // _compute.Call(_compute.ElementwiseMultiplyKernels, a.View, b.View, mulBuffer.View);
        // _compute.Call(_compute.ElementwiseDivideKernels, a.View, b, divBuffer);
        // _compute.Call(_compute.ElementwiseMaxKernels, a.View, b, maxBuffer);
        _compute.Add(addBuffer, a, b);
        _compute.Subtract(subBuffer, a, b);
        _compute.ElementwiseMultiply(mulBuffer, a, b);
        _compute.Divide(divBuffer, a, b);
        _compute.Max(maxBuffer, a, b);
        
        _compute.Synchronize(aidx);
        
        Console.WriteLine($"a + b: {string.Join(',', addBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x + realB[i]))}");
        Console.WriteLine($"a - b: {string.Join(',', subBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x - realB[i]))}");
        Console.WriteLine($"a * b: {string.Join(',', mulBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x * realB[i]))}");
        Console.WriteLine($"a / b: {string.Join(',', divBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => x / realB[i]))}");
        Console.WriteLine($"max(a, b): {string.Join(',', maxBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select((x, i) => Math.Max(x, realB[i])))}");

        using var expBuffer = _compute.Get(aidx, 4);
        using var logBuffer = _compute.Get(aidx, 4);
        using var sqrtBuffer = _compute.Get(aidx, 4);
        using var absBuffer = _compute.Get(aidx, 4);
        using var negateBuffer = _compute.Get(aidx, 4);
        
        // _compute.Call(_compute.ElementwiseExpKernels, a.View, expBuffer.View);
        // _compute.Call(_compute.ElementwiseLogKernels, a.View, logBuffer.View);
        // _compute.Call(_compute.ElementwiseSqrtKernels, a.View, sqrtBuffer.View);
        // _compute.Call(_compute.ElementwiseAbsKernels, a.View, absBuffer.View);
        // _compute.Call(_compute.ElementwiseNegateKernels, a.View, negateBuffer.View);
        _compute.Exp(expBuffer, a);
        _compute.Log(logBuffer, a);
        _compute.Sqrt(sqrtBuffer, a);
        _compute.Abs(absBuffer, a);
        _compute.Negate(negateBuffer, a);
        
        _compute.Synchronize(aidx);
        Console.WriteLine($"exp(a): {string.Join(',', expBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Exp(x)))}");
        Console.WriteLine($"log(a): {string.Join(',', logBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Log(x)))}");
        Console.WriteLine($"sqrt(a): {string.Join(',', sqrtBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => (float)Math.Sqrt(x)))}");
        Console.WriteLine($"abs(a): {string.Join(',', absBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(Math.Abs))}");
        Console.WriteLine($"-a: {string.Join(',', negateBuffer.GetAsArray1D())} vs {string.Join(',', realA.Select(x => -x))}");
        
        var dot = realA.Select((x, i) => x * realB[i]).Sum();
        using var dotBuffer = _compute.Dot(a, b);
        _compute.Synchronize(aidx);
        Console.WriteLine($"dot(a, b): {dot} vs {dotBuffer.GetAsArray1D()[0]}");
        
        _compute.ReleaseAccelerator(aidx);
    }

    public static void MemoryTest(bool gpu)
    {
        int aidx = _compute.RequestAccelerator(gpu);
        
        var a = new ScalarValue(1, aidx);
        var b = new ScalarValue(2, aidx);
        var c = a + b;
        Console.WriteLine(a.Data.GetHashCode());
        Console.WriteLine(b.Data.GetHashCode());
        Console.WriteLine(c.Data.GetHashCode());
        a.Dispose();
        b.Dispose();
        c.Dispose();
        _compute.Synchronize(aidx);

        var d = new ScalarValue(3, aidx);
        var e = d * c;
        Console.WriteLine(d.Data.GetHashCode());
        Console.WriteLine(e.Data.GetHashCode());
        d.Dispose();
        e.Dispose();
        _compute.Synchronize(aidx);

        var f = _compute.Get(aidx, 1);
        Console.WriteLine(f.GetAsArray1D()[0]);
        Console.WriteLine(f.GetHashCode());
        
        _compute.ReleaseAccelerator(aidx);
    }
}