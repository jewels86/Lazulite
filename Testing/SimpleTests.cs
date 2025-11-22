using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Formats.Tar;
using System.Xml;
using ILGPU;
using ILGPU.Runtime;
using Lazulite;
using Lazulite.Values;

namespace Testing;

public static class SimpleTests
{
    private static int _aidx = -1;

    public static void FillTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = new();

        sw.Start();
        for (int i = 0; i < 20000; i++)
        {
            var buffer = Compute.GetTemp(_aidx, 5000);
            Compute.Call(_aidx, Compute.FillKernels, buffer.View, 1);
        }
        sw.Stop();
        Compute.Synchronize(_aidx);
        Console.WriteLine(sw.ElapsedMilliseconds);

        sw.Restart();
        for (int i = 0; i < 20000; i++)
        {
            var buffer = Compute.GetTemp(_aidx, 5000);
            Compute.Call(_aidx, Compute.FillKernels, buffer.View, 1);
        }
        sw.Stop();
        Console.WriteLine(sw.ElapsedMilliseconds);
        Compute.Synchronize(_aidx);
        Compute.ReleaseAccelerator(_aidx);
    }
    
    public static void SimpleMathTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = new();

        double[][,] matrices = new double[200][,];
        for (int i = 0; i < 200; i++)
        {
            double[,] matrix = new double[1000, 1000];
            for (int j = 0; j < 1000; j++) 
            for (int k = 0; k < 1000; k++) matrix[j, k] = j + k;
            matrices[i] = matrix;
        }

        MatrixValue[] buffers = new MatrixValue[200];
        MatrixValue[] results = new MatrixValue[200];
        for (int i = 0; i < 200; i++) buffers[i] = new(matrices[i], _aidx);
        for (int i = 0; i < 200; i++) results[i] = new(new double[1000, 1000], _aidx);

        sw.Start();
        for (int i = 0; i < 200; i++) 
            Compute.Call(_aidx, Compute.ElementwiseMultiplyKernels, buffers[i].Data.View, buffers[i].Data.View, results[i].Data.View);
        Compute.Synchronize(_aidx);
        sw.Stop();
        
        Compute.ClearAll();
        Compute.ReleaseAccelerator(_aidx);
        
        Console.WriteLine(sw.ElapsedMilliseconds);
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
        
        Compute.Synchronize(_aidx);
        Compute.ClearAll();
        Compute.ReleaseAccelerator(_aidx);
    }
    
    public static void PhysicsTest(bool gpu)
    {
        _aidx = Compute.RequestAccelerator(gpu);
        Console.WriteLine(Compute.IsGpuAccelerator(_aidx) ? "GPU accelerator" : "CPU accelerator");
        Stopwatch sw = new();

        int finalT = 100; // end time
        double dt = 0.01; // time step size
        int n = 1000; // number of bodies
        double G = 6.674e-11; // gravitational constant
        Random random = new();

        double[] positions = new double[n * 3];
        double[] velocities = new double[n * 3];
        double[] masses = new double[n];
        double[] forces = new double[n * 3];
        
        Func<double, double, double> randomDouble = (min, max) => random.NextDouble() * (max - min) + min;

        for (int i = 0; i < n; i++)
        {
            (positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]) = (randomDouble(-10, 10), randomDouble(-10, 10), randomDouble(-10, 10));
            (velocities[i * 3], velocities[i * 3 + 1], velocities[i * 3 + 2]) = (0, 0, 0);
            masses[i] = randomDouble(1, 100);
            (forces[i * 3], forces[i * 3 + 1], forces[i * 3 + 2]) = (0, 0, 0);
        }

        MemoryBuffer1D<double, Stride1D.Dense> positionsBuffer = Compute.GetTemp(_aidx, n * 3);
        MemoryBuffer1D<double, Stride1D.Dense> velocitiesBuffer = Compute.GetTemp(_aidx, n * 3);
        MemoryBuffer1D<double, Stride1D.Dense> massesBuffer = Compute.GetTemp(_aidx, n);
        MemoryBuffer1D<double, Stride1D.Dense> forcesBuffer = Compute.GetTemp(_aidx, n * 3);
        positionsBuffer.CopyFromCPU(positions);
        velocitiesBuffer.CopyFromCPU(velocities);
        massesBuffer.CopyFromCPU(masses);
        forcesBuffer.CopyFromCPU(forces);
        var extent = new Index1D(n);

        Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, double, int> gravityKernel =
            (index, rs, ms, fs, g, total) =>
            {
                int i = index.X;
                var (fx, fy, fz) = (0.0, 0.0, 0.0);
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
        Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, double>
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
        for (double t = 0; t < finalT; t += dt)
        {
            Compute.Call(_aidx, gravityKernels, extent, positionsBuffer.View, massesBuffer.View, forcesBuffer.View, G, n);
            Compute.Call(_aidx, eulerKernels, extent, positionsBuffer.View, velocitiesBuffer.View, forcesBuffer.View, massesBuffer.View, dt);
        }
        Compute.Synchronize(_aidx);
        sw.Stop();
        
        Compute.ClearAll();
        Compute.ReleaseAccelerator(_aidx);
        
        Console.WriteLine($"Total timesteps: {finalT / dt}.");
        Console.WriteLine($"Total bodies: {n}");
        Console.WriteLine($"Bodies processed per ms: {(finalT / dt) * n / sw.ElapsedMilliseconds:F2}.)");
        Console.WriteLine($"Elapsed time: {sw.ElapsedMilliseconds} ms ({sw.ElapsedMilliseconds / (finalT/dt):F2} ms/timestep, or {(finalT/dt) / sw.ElapsedMilliseconds:F2} timesteps/ms).");
    }
}