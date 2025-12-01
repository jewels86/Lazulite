using System.Collections.Concurrent;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Testing;

public static class SimpleTests
{
    private static Compute _compute => Compute.Instance;
    private static int _aidx = -1;
    private static Random _random = new();

    public static void FillTest(bool gpu)
    {
        _aidx = _compute.RequestAccelerator(gpu);
        Console.WriteLine(_compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = new();

        var (n, size) = (20000, 5000);

        sw.Start();
        for (int i = 0; i < n; i++)
        {
            var buffer = _compute.Get(_aidx, size);
            _compute.Call(_compute.FillKernels, buffer.View, 1);
            _compute.Return(buffer);
        }
        sw.Stop();
        _compute.Synchronize(_aidx);
        Console.WriteLine($"Filled {n} vectors of size {size} in {sw.ElapsedMilliseconds} (no pre-allocations - {n * size} elements) where allocations are done in-loop.");

        sw.Restart();
        for (int i = 0; i < n; i++)
        {
            var buffer = _compute.Get(_aidx, size);
            _compute.Call(_compute.FillKernels, buffer.View, 1);
            _compute.Return(buffer);
        }
        sw.Stop();
        Console.WriteLine($"Filled {n} vectors of size {size} in {sw.ElapsedMilliseconds} (pre-allocations - {n * size} elements) where allocations are done in-loop.");
        _compute.Synchronize(_aidx);
        _compute.ReleaseAccelerator(_aidx);
    }
    
    public static void SimpleMathTest(bool gpu)
    {
        _aidx = _compute.RequestAccelerator(gpu);
        Console.WriteLine(_compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
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
            _compute.Call(_compute.ElementwiseMultiplyKernels, buffers[i].Data.View, buffers[i].Data.View, results[i].Data.View);
        _compute.Synchronize(_aidx);
        sw.Stop();
        
        _compute.ReleaseAccelerator(_aidx);
        
        Console.WriteLine($"Squared each element of {n} {size}x{size} matrices ({size * size * n} elements) in {sw.ElapsedMilliseconds} ms.");
    }

    public static void ScalarTest(bool gpu)
    {
        _aidx = _compute.RequestAccelerator(gpu);
        Console.WriteLine(_compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        
        ScalarValue a = new(1, _aidx);
        ScalarValue b = new(2, _aidx);
        ScalarValue c = new(3, _aidx);
        
        var d = a + b;
        var e = d * c;
        var f = e - a;
        // this is bad- we allocated 3 values but only one is output
        
        Console.WriteLine(f);
        
        f = _compute.BinaryCallChain(a, 
            (_compute.ElementwiseAddKernels, b),
            (_compute.ElementwiseMultiplyKernels, c),
            (_compute.ElementwiseSubtractKernels, a)).AsScalar();
        // this is much better- we only allocate 1 value, f
        
        Console.WriteLine(f);
        
        var addMultSub = _compute.Load((i, a_, b_, c_, r) => r[i] = ((a_[i] + b_[i]) * c_[i]) - a_[i]);
        _compute.Call(_aidx, addMultSub, a.Data.IntExtent, a.Data.View, b.Data.View, c.Data.View, f.Data.View);
        // this is even better- we only allocate 1 value, f, and we can fuse the operations into one kernel call
        Console.WriteLine(f);
        
        _compute.Synchronize(_aidx);
        _compute.ReleaseAccelerator(_aidx);
    }
    
    public static void PhysicsTest(bool gpu)
    {
        _aidx = _compute.RequestAccelerator(gpu);
        Console.WriteLine(_compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
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

        MemoryBuffer1D<float, Stride1D.Dense> positionsBuffer = _compute.Get(_aidx, n * 3);
        MemoryBuffer1D<float, Stride1D.Dense> velocitiesBuffer = _compute.Get(_aidx, n * 3);
        MemoryBuffer1D<float, Stride1D.Dense> massesBuffer = _compute.Get(_aidx, n);
        MemoryBuffer1D<float, Stride1D.Dense> forcesBuffer = _compute.Get(_aidx, n * 3);
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
                var (rx, ry, rz) = KernelProgramming.Vector3Get(rs, i);

                for (int j = 0; j < total; j++)
                {
                    if (i == j) continue;
                    
                    var (jrx, jry, jrz) = KernelProgramming.Vector3Get(rs, j);
                    var (dx, dy, dz) = KernelProgramming.Vector3Subtract((jrx, jry, jrz), (rx, ry, rz));

                    var r2 = KernelProgramming.Vector3Magnitude2((dx, dy, dz));
                    var f = g * ms[i] * ms[j] / r2;

                    var (dfx, dfy, dfz) = KernelProgramming.Vector3Multiply((dx, dy, dz), f);
                    (fx, fy, fz) = KernelProgramming.Vector3Add((fx, fy, fz), (dfx, dfy, dfz));
                }

                (fx, fy, fz) = KernelProgramming.Vector3Divide((fx, fy, fz), ms[i]);
                KernelProgramming.Vector3Set(fs, i, (fx, fy, fz));
            };
        Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>
            eulerKernel =
                (index, rs, vs, fs, ms, dt_) =>
                {
                    int i = index.X;
                    var (x, y, z) = KernelProgramming.Vector3Get(rs, i);
                    var (vx, vy, vz) = KernelProgramming.Vector3Get(vs, i);
                    var (fx, fy, fz) = KernelProgramming.Vector3Get(fs, i);

                    var (ax, ay, az) = KernelProgramming.Vector3Divide((fx, fy, fz), ms[i]);
                    var (dvx, dvy, dvz) = KernelProgramming.Vector3Multiply((ax, ay, az), dt_);
                    var (vx2, vy2, vz2) = KernelProgramming.Vector3Add((vx, vy, vz), (dvx, dvy, dvz));
                    KernelProgramming.Vector3Set(vs, i, (vx2, vy2, vz2));

                    var (drx, dry, drz) = KernelProgramming.Vector3Multiply((vx2, vy2, vz2), dt_);
                    var (x2, y2, z2) = KernelProgramming.Vector3Add((x, y, z), (drx, dry, drz));
                    KernelProgramming.Vector3Set(rs, i, (x2, y2, z2));
                };

        var gravityKernels = _compute.Load(gravityKernel);
        var eulerKernels = _compute.Load(eulerKernel);
        
        sw.Start();
        for (float t = 0; t < finalT; t += dt)
        {
            _compute.Call(_aidx, gravityKernels, extent, positionsBuffer.View, massesBuffer.View, forcesBuffer.View, G, n);
            _compute.Call(_aidx, eulerKernels, extent, positionsBuffer.View, velocitiesBuffer.View, forcesBuffer.View, massesBuffer.View, dt);
        }
        _compute.Synchronize(_aidx);
        sw.Stop();
        
        _compute.Return(positionsBuffer, velocitiesBuffer, massesBuffer, forcesBuffer);
        
        _compute.ReleaseAccelerator(_aidx);
        
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

        int aidx = _compute.RequestAccelerator(gpu);
        Stopwatch sw = new();
        
        Console.WriteLine($"Starting processing on accelerator {aidx} ({_compute.Accelerators[aidx].Name}).");
        sw.Start();
        foreach (var (a, b) in workQueue)
        {
            var aBuffer = _compute.Get(aidx, mk);
            var bBuffer = _compute.Get(aidx, kn);
            var resultBuffer = _compute.Get(aidx, mn);
            aBuffer.CopyFromCPU(a);
            bBuffer.CopyFromCPU(b);
            _compute.MatrixMultiply(aBuffer, bBuffer, resultBuffer, m, k, k, n, noCuBlas: !cublas);
            results.Add(resultBuffer);
            _compute.Return(aBuffer, bBuffer);
        }
        _compute.Synchronize(aidx);
        sw.Stop();
        
        Console.WriteLine($"Processed {totalBatches} batches of {m}x{k}x{n} matrix multiplies in: {sw.ElapsedMilliseconds} ms.");
        
        _compute.Return(results.ToArray());
        _compute.Clear(aidx);
        _compute.ReleaseAccelerator(aidx);
        results.Clear();
        
        var gpuIndices = _compute.Accelerators.Values
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
            Console.WriteLine($"Starting parallel processing on accelerator {aidx_} ({_compute.Accelerators[aidx_].Name}).");
            tasks[i] = Task.Run(() =>
            {
                while (workQueue.TryDequeue(out var tuple))
                {
                    var aBuffer = _compute.Get(aidx_, mk);
                    var bBuffer = _compute.Get(aidx_, kn);
                    var resultBuffer = _compute.Get(aidx_, mn);
        
                    aBuffer.CopyFromCPU(tuple.a);
                    bBuffer.CopyFromCPU(tuple.b);
                    _compute.MatrixMultiply(aBuffer, bBuffer, resultBuffer, m, k, k, n, noCuBlas:!cublas);
        
                    results.Add(resultBuffer);
                    _compute.Return(aBuffer, bBuffer);
                }
                _compute.Synchronize(aidx_);
                Console.WriteLine($"{aidx_} finished processing!");
            });
        }
        Task.WaitAll(tasks);
        sw.Stop();
        
        _compute.Return(results.ToArray());
        _compute.ReleaseAccelerator(aidx);
        results.Clear();
        Console.WriteLine($"Processed {totalBatches} batches of {m}x{k}x{n} matrix multiplies in: {sw.ElapsedMilliseconds} ms.");
    }

    public static void BigMatMulTest(bool gpu)
    {
        _aidx = _compute.RequestAccelerator(gpu);
        Console.WriteLine(_compute.IsGpuAccelerator(_aidx) ? $"GPU accelerator {_aidx}" : "CPU accelerator");
        Stopwatch sw = new();
        
        int m = 10000;
        int k = 10000;
        int n = 10000;
        int mk = m * k;
        int kn = k * n;
        int mn = m * n;
        
        float[,] a = RandomMatrix(m, k);
        float[,] b = RandomMatrix(k, n);
        MemoryBuffer1D<float, Stride1D.Dense> aBuffer = _compute.Get(_aidx, mk);
        MemoryBuffer1D<float, Stride1D.Dense> bBuffer = _compute.Get(_aidx, kn);
        MemoryBuffer1D<float, Stride1D.Dense> resultBuffer = _compute.Get(_aidx, mn);
        aBuffer.CopyFromCPU(MatrixProxy.Roll(a));
        bBuffer.CopyFromCPU(MatrixProxy.Roll(b));
        
        Console.WriteLine("Starting processing...");
        sw.Start();
        _compute.MatrixMultiply(aBuffer, bBuffer, resultBuffer, m, k, k, n, noCuBlas: false);
        Console.WriteLine("Finished processing!");
        _compute.Synchronize(_aidx);
        sw.Stop();
        
        Console.WriteLine($"{m}x{k}x{n} matrix multiply finished in: {sw.ElapsedMilliseconds} ms.");
        _compute.Return(resultBuffer, aBuffer, bBuffer);
        _compute.ReleaseAccelerator(_aidx);
    }

    public static void PoolTest(bool gpu)
    {
        _aidx = _compute.RequestAccelerator(gpu);
        Console.WriteLine(_compute.IsGpuAccelerator(_aidx) ? $"GPU accelerator {_aidx}" : "CPU accelerator");
        Stopwatch sw = new();

        var (n, size) = (1000, 10000);
        var kernels = _compute.Load((i, a, b, r) => r[i] += (a[i] * a[i] - b[i] * b[i]) / (a[i] * a[i] + b[i] * b[i]));
        var result = _compute.Get(_aidx, size);
        
        sw.Start();
        for (int i = 0; i < n; i++)
        {
            var bufferA = _compute.Get(_aidx, size).Set(RandomVector(size));
            var bufferB = _compute.Get(_aidx, size).Set(RandomVector(size));
            
            _compute.Call(kernels, bufferA, bufferB, result);
            _compute.Return(bufferA, bufferB);
        }
        _compute.Synchronize(_aidx);
        sw.Stop();
        
        Console.WriteLine($"Processed {n} batches in {sw.ElapsedMilliseconds} ms.");
        _compute.Return(result);
        _compute.ReleaseAccelerator(_aidx);
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

    public static float[] RandomVector(int size)
    {
        float[] vector = new float[size];
        for (int i = 0; i < size; i++) vector[i] = (float)_random.NextDouble();
        return vector;
    }
}