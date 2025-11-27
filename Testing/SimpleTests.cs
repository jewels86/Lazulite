using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Formats.Tar;
using System.Xml;
using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Testing;

public static class SimpleTests
{
    private static int _aidx = -1;
    private static Random _random = new();

    public static void FillTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = new();

        var (n, size) = (20000, 5000);

        sw.Start();
        for (int i = 0; i < n; i++)
        {
            var buffer = Compute.GetTemp(_aidx, size);
            Compute.Call(_aidx, Compute.FillKernels, buffer.View, 1);
        }
        sw.Stop();
        Compute.Synchronize(_aidx);
        Console.WriteLine($"Filled {n} vectors of size {size} in {sw.ElapsedMilliseconds} (no pre-allocations - {n * size} elements) where allocations are done in-loop.");

        sw.Restart();
        for (int i = 0; i < n; i++)
        {
            var buffer = Compute.GetTemp(_aidx, size);
            Compute.Call(_aidx, Compute.FillKernels, buffer.View, 1);
        }
        sw.Stop();
        Console.WriteLine($"Filled {n} vectors of size {size} in {sw.ElapsedMilliseconds} (pre-allocations - {n * size} elements) where allocations are done in-loop.");
        Compute.Synchronize(_aidx);
        Compute.ReleaseAccelerator(_aidx);
    }
    
    public static void SimpleMathTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = new();
        
        var (n, size) = (200, 1000);

        float[][,] matrices = new float[n][,];
        for (int i = 0; i < n; i++)
        {
            float[,] matrix = new float[size, size];
            for (int j = 0; j < size; j++) 
            for (int k = 0; k < size; k++) matrix[j, k] = j + k;
            matrices[i] = matrix;
        }

        MatrixValue[] buffers = new MatrixValue[n];
        MatrixValue[] results = new MatrixValue[n];
        for (int i = 0; i < n; i++) buffers[i] = new(matrices[i], _aidx);
        for (int i = 0; i < n; i++) results[i] = new(new float[size, size], _aidx);

        sw.Start();
        for (int i = 0; i < n; i++) 
            Compute.Call(_aidx, Compute.ElementwiseMultiplyKernels, buffers[i].Data.View, buffers[i].Data.View, results[i].Data.View);
        Compute.Synchronize(_aidx);
        sw.Stop();
        
        Compute.ReleaseAccelerator(_aidx);
        
        Console.WriteLine($"Squared each element of {n} {size}x{size} matrices ({size * size * n} elements) in {sw.ElapsedMilliseconds} ms.");
    }

    public static void ScalarTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        
        ScalarValue a = new(1, _aidx);
        ScalarValue b = new(2, _aidx);
        ScalarValue c = new(3, _aidx);
        
        var d = a + b;
        var e = d * c;
        var f = e - a;
        // this is bad- we allocated 3 values but only one is output
        
        Console.WriteLine(f);
        
        f = Compute.BinaryCallChain(a, 
            (Compute.ElementwiseAddKernels, b),
            (Compute.ElementwiseMultiplyKernels, c),
            (Compute.ElementwiseSubtractKernels, a)).AsScalar();
        // this is much better- we only allocate 1 value, f
        
        Console.WriteLine(f);
        
        var addMultSub = Compute.Load((i, a_, b_, c_, r) => r[i] = ((a_[i] + b_[i]) * c_[i]) - a_[i]);
        Compute.Call(_aidx, addMultSub, a.Data.IntExtent, a.Data.View, b.Data.View, c.Data.View, f.Data.View);
        // this is even better- we only allocate 1 value, f, and we can fuse the operations into one kernel call
        Console.WriteLine(f);
        
        Compute.Synchronize(_aidx);
        Compute.ReleaseAccelerator(_aidx);
    }
    
    public static void PhysicsTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = new();

        int finalT = 100; // end time
        float dt = 0.01f; // time step size
        int n = 1000; // number of bodies
        float G = 6.674e-11f; // gravitational constant
        Random random = new();

        float[] positions = new float[n * 3];
        float[] velocities = new float[n * 3];
        float[] masses = new float[n];
        float[] forces = new float[n * 3];
        
        Func<float, float, float> randomDouble = (min, max) => (float)random.NextDouble() * (max - min) + min;

        for (int i = 0; i < n; i++)
        {
            (positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]) = (randomDouble(-10, 10), randomDouble(-10, 10), randomDouble(-10, 10));
            (velocities[i * 3], velocities[i * 3 + 1], velocities[i * 3 + 2]) = (0, 0, 0);
            masses[i] = randomDouble(1, 100);
            (forces[i * 3], forces[i * 3 + 1], forces[i * 3 + 2]) = (0, 0, 0);
        }

        MemoryBuffer1D<float, Stride1D.Dense> positionsBuffer = Compute.GetTemp(_aidx, n * 3);
        MemoryBuffer1D<float, Stride1D.Dense> velocitiesBuffer = Compute.GetTemp(_aidx, n * 3);
        MemoryBuffer1D<float, Stride1D.Dense> massesBuffer = Compute.GetTemp(_aidx, n);
        MemoryBuffer1D<float, Stride1D.Dense> forcesBuffer = Compute.GetTemp(_aidx, n * 3);
        positionsBuffer.CopyFromCPU(positions);
        velocitiesBuffer.CopyFromCPU(velocities);
        massesBuffer.CopyFromCPU(masses);
        forcesBuffer.CopyFromCPU(forces);
        var extent = new Index1D(n);

        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float, int> gravityKernel =
            (index, rs, ms, fs, g, total) =>
            {
                int i = index.X;
                var (fx, fy, fz) = (0f, 0f, 0f);
                var (rx, ry, rz) = KernelProgramming.GetVector3(rs, i);

                for (int j = 0; j < total; j++)
                {
                    if (i == j) continue;
                    
                    var (jrx, jry, jrz) = KernelProgramming.GetVector3(rs, j);
                    var (dx, dy, dz) = KernelProgramming.SubtractVector3((jrx, jry, jrz), (rx, ry, rz));

                    var r2 = KernelProgramming.Magnitude2Vector3((dx, dy, dz));
                    var f = g * ms[i] * ms[j] / r2;

                    var (dfx, dfy, dfz) = KernelProgramming.MultiplyVector3((dx, dy, dz), f);
                    (fx, fy, fz) = KernelProgramming.AddVector3((fx, fy, fz), (dfx, dfy, dfz));
                }

                (fx, fy, fz) = KernelProgramming.DivideVector3((fx, fy, fz), ms[i]);
                KernelProgramming.SetVector3(fs, i, (fx, fy, fz));
            };
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>
            eulerKernel =
                (index, rs, vs, fs, ms, dt_) =>
                {
                    int i = index.X;
                    var (x, y, z) = KernelProgramming.GetVector3(rs, i);
                    var (vx, vy, vz) = KernelProgramming.GetVector3(vs, i);
                    var (fx, fy, fz) = KernelProgramming.GetVector3(fs, i);

                    var (ax, ay, az) = KernelProgramming.DivideVector3((fx, fy, fz), ms[i]);
                    var (dvx, dvy, dvz) = KernelProgramming.MultiplyVector3((ax, ay, az), dt_);
                    var (vx2, vy2, vz2) = KernelProgramming.AddVector3((vx, vy, vz), (dvx, dvy, dvz));
                    KernelProgramming.SetVector3(vs, i, (vx2, vy2, vz2));

                    var (drx, dry, drz) = KernelProgramming.MultiplyVector3((vx2, vy2, vz2), dt_);
                    var (x2, y2, z2) = KernelProgramming.AddVector3((x, y, z), (drx, dry, drz));
                    KernelProgramming.SetVector3(rs, i, (x2, y2, z2));
                };

        var gravityKernels = Compute.Load(gravityKernel);
        var eulerKernels = Compute.Load(eulerKernel);
        
        sw.Start();
        for (float t = 0; t < finalT; t += dt)
        {
            Compute.Call(_aidx, gravityKernels, extent, positionsBuffer.View, massesBuffer.View, forcesBuffer.View, G, n);
            Compute.Call(_aidx, eulerKernels, extent, positionsBuffer.View, velocitiesBuffer.View, forcesBuffer.View, massesBuffer.View, dt);
        }
        Compute.Synchronize(_aidx);
        sw.Stop();
        
        Compute.ReleaseAccelerator(_aidx);
        
        Console.WriteLine($"Total timesteps: {finalT / dt}");
        Console.WriteLine($"Total bodies: {n}");
        Console.WriteLine($"Bodies processed per ms: {(finalT / dt) * n / sw.ElapsedMilliseconds:F2}");
        Console.WriteLine($"Elapsed time: {sw.ElapsedMilliseconds} ms ({sw.ElapsedMilliseconds / (finalT/dt):F2} ms/timestep, or {(finalT/dt) / sw.ElapsedMilliseconds:F2} timesteps/ms)");
    }
    
    public static void ParallelProcessingTest(bool gpu, bool cublas = true)
    {
        Console.WriteLine(cublas ? "Using cuBLAS" : "Using no-BLAS");
        int totalBatches = 500;
        var (m, k, n) = (256, 512, 256);
        int mk = m * k;
        int kn = k * n;
        int mn = m * n;
        Index1D extent = new(mn);

        ConcurrentQueue<(float[] a, float[] b)> workQueue = [];
        ConcurrentBag<MemoryBuffer1D<float, Stride1D.Dense>> results = new();
        Random random = new();
        
        Console.WriteLine("Generating matrices...");
        for (int i = 0; i < totalBatches; i++) workQueue.Enqueue((
            MatrixProxy.Roll(RandomMatrix(m, k)), 
            MatrixProxy.Roll(RandomMatrix(k, n))));

        int aidx = Compute.RequestAccelerator(gpu);
        Stopwatch sw = new();
        
        Console.WriteLine($"Starting processing on accelerator {aidx} ({Compute.Accelerators[aidx].Name}).");
        sw.Start();
        foreach (var (a, b) in workQueue)
        {
            var aBuffer = Compute.GetTemp(aidx, mk);
            var bBuffer = Compute.GetTemp(aidx, kn);
            var resultBuffer = Compute.Get(aidx, mn);
            aBuffer.CopyFromCPU(a);
            bBuffer.CopyFromCPU(b);
            Compute.MatrixMultiply(aBuffer, bBuffer, resultBuffer, m, k, n, !cublas);
            results.Add(resultBuffer);
            Compute.Flush(aidx);
        }
        Compute.Synchronize(aidx);
        sw.Stop();
        
        Console.WriteLine($"Processed {totalBatches} batches of {m}x{k}x{n} matrix multiplies in: {sw.ElapsedMilliseconds} ms.");
        
        Compute.Return(results.ToArray());
        Compute.Clear(aidx);
        Compute.ReleaseAccelerator(aidx);
        results.Clear();
        
        var gpuIndices = Compute.Accelerators
            .Select((acc, idx) => (acc, idx))
            .Where(x => x.acc.AcceleratorType != AcceleratorType.CPU)
            .Select(x => x.idx)
            .ToList(); // you can run this test with a CPU as well, but the tail will bite you unless you have a lottt of matrices

        
        Console.WriteLine("Starting parallel processing...");
        sw.Restart();
        Task[] tasks = new Task[gpuIndices.Count];
        for (int i = 0; i < gpuIndices.Count; i++)
        {
            int aidx_ = gpuIndices[i];
            Console.WriteLine($"Starting parallel processing on accelerator {aidx_} ({Compute.Accelerators[aidx_].Name}).");
            tasks[i] = Task.Run(() =>
            {
                while (workQueue.TryDequeue(out var tuple))
                {
                    var aBuffer = Compute.GetTemp(aidx_, mk);
                    var bBuffer = Compute.GetTemp(aidx_, kn);
                    var resultBuffer = Compute.Get(aidx_, mn);
        
                    aBuffer.CopyFromCPU(tuple.a);
                    bBuffer.CopyFromCPU(tuple.b);
                    Compute.MatrixMultiply(aBuffer, bBuffer, resultBuffer, m, k, n, !cublas);
        
                    results.Add(resultBuffer);
                    Compute.Flush(aidx_);
                }
                Compute.Synchronize(aidx_);
                Console.WriteLine($"{aidx_} finished processing!");
            });
        }
        Task.WaitAll(tasks);
        sw.Stop();
        
        Compute.Return(results.ToArray());
        Compute.ReleaseAccelerator(aidx);
        results.Clear();
        Console.WriteLine($"Processed {totalBatches} batches of {m}x{k}x{n} matrix multiplies in: {sw.ElapsedMilliseconds} ms.");
    }

    public static void BigMatMulTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? $"GPU accelerator {_aidx}" : "CPU accelerator");
        Stopwatch sw = new();
        
        int m = 10000;
        int k = 10000;
        int n = 10000;
        int mk = m * k;
        int kn = k * n;
        int mn = m * n;
        
        float[,] a = RandomMatrix(m, k);
        float[,] b = RandomMatrix(k, n);
        MemoryBuffer1D<float, Stride1D.Dense> aBuffer = Compute.GetTemp(_aidx, mk);
        MemoryBuffer1D<float, Stride1D.Dense> bBuffer = Compute.GetTemp(_aidx, kn);
        MemoryBuffer1D<float, Stride1D.Dense> resultBuffer = Compute.Get(_aidx, mn);
        aBuffer.CopyFromCPU(MatrixProxy.Roll(a));
        bBuffer.CopyFromCPU(MatrixProxy.Roll(b));
        
        Console.WriteLine("Starting processing...");
        sw.Start();
        Compute.MatrixMultiply(aBuffer, bBuffer, resultBuffer, m, k, n, false);
        Console.WriteLine("Finished processing!");
        Compute.Synchronize(_aidx);
        sw.Stop();
        
        Console.WriteLine($"{m}x{k}x{n} matrix multiply finished in: {sw.ElapsedMilliseconds} ms.");
    }
    
    public static float[,] RandomMatrix(int rows, int cols)
    {
        float[,] matrix = new float[rows, cols];
        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++) matrix[i, j] = (float)_random.NextDouble();
        });
        return matrix;
    }
}