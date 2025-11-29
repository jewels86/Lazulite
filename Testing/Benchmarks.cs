using System.Diagnostics;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Testing;

public static class Benchmarks
{
    private static Compute compute => Compute.Instance;
    private static int aidx = -1;
    private static readonly Random random = new();
    private static readonly HashSet<int> printed = [];

    public static void PrintInfo(int idx)
    {
        if (!printed.Add(idx)) return;
        StringWriter sw = new();
        compute.Accelerators[idx].PrintInformation(sw);
        Console.WriteLine($"Accelerator {idx} info: \n" + sw);
    }

    #region Simple Benchmarks
    public static void SquareBenchmark1(bool gpu, int l = 1000, int w = 1000, int n = 200)
    {
        aidx = compute.RequestAccelerator(gpu);
        string acceleratorType = gpu ? "GPU" : "CPU";
        PrintInfo(aidx);
        Console.WriteLine($"Square Benchmark starting using accelerator {aidx} ({compute.Accelerators[aidx].Name} - {acceleratorType} accelerator).");
        
        Console.WriteLine($"Creating {n} matrices of size {l}x{w} ({n * l * w:E2} elements total) and storing them in memory buffers on accelerator {aidx}...");
        MatrixValue[] matrices = new MatrixValue[n];
        var results = compute.Get(aidx, n, l * w);
        for (int i = 0; i < n; i++) matrices[i] = new(RandomMatrix(l, w), aidx);
        
        Console.WriteLine($"Multiplying elementwise with {n} matrices...");
        
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < n; i++) compute.Call(compute.ElementwiseMultiplyKernels, matrices[i], matrices[i], results[i]);
        compute.Synchronize(aidx);
        sw.Stop();
        Console.WriteLine($"Elementwise multiplication took {sw.ElapsedMilliseconds} ms.");
        
        compute.ReleaseAccelerator(aidx);
        foreach (var matrix in matrices) matrix.Dispose();
        foreach (var result in results) result.Dispose();
    }

    public static void SquareBenchmark2(bool gpu, int l = 1000, int w = 1000, int n = 200)
    {
        aidx = compute.RequestAccelerator(gpu);
        string acceleratorType = gpu ? "GPU" : "CPU";
        PrintInfo(aidx);
        Console.WriteLine($"Square Benchmark (with matrix values) starting using accelerator {aidx} ({compute.Accelerators[aidx].Name} - {acceleratorType} accelerator).");
        
        Console.WriteLine($"Creating {n} matrices of size {l}x{w} ({n * l * w:E2} elements total).");
        MatrixValue[] matrices = new MatrixValue[n];
        MatrixValue[] results = new MatrixValue[n];
        for (int i = 0; i < n; i++) matrices[i] = new(RandomMatrix(l, w), aidx);
        
        Console.WriteLine($"Multiplying elementwise with {n} matrices...");
        
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < n; i++) results[i] = matrices[i] * matrices[i];
        compute.Synchronize(aidx);
        sw.Stop();
        Console.WriteLine($"Elementwise multiplication took {sw.ElapsedMilliseconds} ms.");
        
        compute.ReleaseAccelerator(aidx);
        foreach (var matrix in matrices) matrix.Dispose();
        foreach (var result in results) result.Dispose();
    }

    public static void MatMulBenchmark(bool gpu, int m = 1000, int n = 1000, int k = 1000, int totalBatches = 100)
    {
        aidx = compute.RequestAccelerator(gpu);
        string acceleratorType = gpu ? "GPU" : "CPU";
        PrintInfo(aidx);
        Console.WriteLine($"Matrix multiplication Benchmark starting using accelerator {aidx} ({compute.Accelerators[aidx].Name} - {acceleratorType} accelerator).");
        Console.WriteLine(m * n > 1e3 ? "This benchmark will use CuBLAS! (m * n > 1e3)" : "This benchmark will use a naive implementation... (m * n < 1e3)");
        
        Console.WriteLine($"Creating {totalBatches} matrices of size {m}x{k} and {k}x{n} ({totalBatches * m * k + totalBatches * k * n:E2} elements total).");
        MatrixValue[,] matrices = new MatrixValue[totalBatches, 2];
        var results = compute.Get(aidx, totalBatches, m * n);
        for (int i = 0; i < totalBatches; i++) matrices[i, 0] = new(RandomMatrix(m, k), aidx);
        for (int i = 0; i < totalBatches; i++) matrices[i, 1] = new(RandomMatrix(k, n), aidx);
        
        Console.WriteLine($"Total result elements: {totalBatches * m * n:E2}.");
        Console.WriteLine($"Matrix multiplying {totalBatches} batches of matrices...");
        
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < totalBatches; i++) compute.MatrixMultiply(matrices[i, 0], matrices[i, 1], results[i], m, k, n);
        compute.Synchronize(aidx);
        sw.Stop();
        
        Console.WriteLine($"Multiplication took {sw.ElapsedMilliseconds} ms.");
        
        compute.ReleaseAccelerator(aidx);
        foreach (var matrix in matrices) matrix.Dispose();
        foreach (var result in results) result.Dispose();
    }
    #endregion

    private static float[,] RandomMatrix(int l, int w)
    {
        float[,] matrix = new float[l, w];
        Parallel.For(0, l, i =>
        {
            for (int j = 0; j < w; j++) matrix[i, j] = (float)random.NextDouble();
        });
        return matrix;
    }
}