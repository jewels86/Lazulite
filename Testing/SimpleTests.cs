using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        sw.Stop();
        
        Compute.Synchronize(_aidx);
        Compute.FlushAll();
        Compute.ReleaseAccelerator(_aidx);
        
        Console.WriteLine(sw.ElapsedMilliseconds);
    }
}