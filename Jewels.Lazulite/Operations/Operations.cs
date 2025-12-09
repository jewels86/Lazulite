using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public partial class Compute
{
    private readonly static Dictionary<int, CuBlas?> _cublasHandles = [];
    
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ApxyKernels { get; private set; } = [];

    public static void InitializeCuBlas()
    {
        foreach (var aidx in Accelerators.Keys) GetCuBlas(aidx);
        ApxyKernels = Load((
            Index1D i,
            ArrayView1D<float, Stride1D.Dense> x,
            ArrayView1D<float, Stride1D.Dense> y,
            float alpha) => y[i] = alpha * x[i] + y[i]);
    }

    public static void CleanupCuBlas()
    {
        foreach (var handle in _cublasHandles.Values) handle?.Dispose();
        _cublasHandles.Clear();
    }

    public static CuBlas? GetCuBlas(int aidx)
    {
        if (_cublasHandles.TryGetValue(aidx, out var blas) || Accelerators[aidx] is not CudaAccelerator cudaAccelerator) return blas;
        try
        {
            blas = new CuBlas(cudaAccelerator);
            _cublasHandles[aidx] = blas;
        }
        catch (Exception) { _cublasHandles[aidx] = null; }
        return blas;
    }
    
    public static void Fill(MemoryBuffer1D<float, Stride1D.Dense> buffer, float value) => Call(FillKernels, buffer, value);
    public static void Zero(MemoryBuffer1D<float, Stride1D.Dense> buffer) => Call(ZeroKernels, buffer);
    public static void Copy(MemoryBuffer1D<float, Stride1D.Dense> dest, MemoryBuffer1D<float, Stride1D.Dense> src) => Call(CopyKernels, dest, src);

    public static void Add(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(AddKernels, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Add(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Add(r, a, b));

    public static void Subtract(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(SubtractKernels, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Subtract(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Subtract(r, a, b));

    public static void ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(ElementwiseMultiplyKernels, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => ElementwiseMultiply(r, a, b));

    public static void Divide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(DivideKernels, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Divide(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Divide(r, a, b));

    public static void Max(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(MaxKernels, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Max(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Max(r, a, b));

    public static void Exp(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(ExpKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Exp(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Exp(r, val));

    public static void Log(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(LogKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Log(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Log(r, val));

    public static void Sqrt(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(SqrtKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sqrt(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Sqrt(r, val));

    public static void Abs(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(AbsKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Abs(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Abs(r, val));

    public static void Negate(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(NegateKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Negate(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Negate(r, val));
    
    public static void Sine(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(SineKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sine(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Encase(val, r => Sine(r, val));
    
    public static void Cosine(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(CosineKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Cosine(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Encase(val, r => Cosine(r, val));
    
    public static void Tangent(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(TangentKernels, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Tangent(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Encase(val, r => Tangent(r, val));

    public static void ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarPowerKernels, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarPower(r, value, scalar));

    public static void ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarMultiplyKernels, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarMultiply(r, value, scalar));

    public static void ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarDivideKernels, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarDivide(r, value, scalar));

    public static void ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarMaxKernels, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarMax(r, value, scalar));
    
    public static void FloatPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatPowerKernels, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatPower(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatPower(r, value, scalar));

    public static void FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatMultiplyKernels, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatMultiply(r, value, scalar));

    public static void FloatMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatMaxKernels, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMax(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatMax(r, value, scalar));

    public static MemoryBuffer1D<float, Stride1D.Dense> Sum(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val.AcceleratorIndex(), 1, r => Sum(r, val));
    
    public static MemoryBuffer1D<float, Stride1D.Dense> Dot(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Encase(a.AcceleratorIndex(), 1, r => Dot(r, a, b));
}