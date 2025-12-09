using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Jewels.Lazulite;

public partial class Compute
{
    private readonly Dictionary<int, CuBlas?> _cublasHandles = [];
    
    public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ApxyKernels { get; private set; } = [];

    public void InitializeCuBlas()
    {
        foreach (var aidx in Accelerators.Keys) GetCuBlas(aidx);
        ApxyKernels = Load((
            Index1D i,
            ArrayView1D<float, Stride1D.Dense> x,
            ArrayView1D<float, Stride1D.Dense> y,
            float alpha) => y[i] = alpha * x[i] + y[i]);
    }

    public void CleanupCuBlas()
    {
        foreach (var handle in _cublasHandles.Values) handle?.Dispose();
        _cublasHandles.Clear();
    }

    public CuBlas? GetCuBlas(int aidx)
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
    
    public void Fill(MemoryBuffer1D<float, Stride1D.Dense> buffer, float value) => Call(FillKernels, buffer, value);
    public void Zero(MemoryBuffer1D<float, Stride1D.Dense> buffer) => Call(ZeroKernels, buffer);
    public void Copy(MemoryBuffer1D<float, Stride1D.Dense> dest, MemoryBuffer1D<float, Stride1D.Dense> src) => Call(CopyKernels, dest, src);

    public void Add(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(AddKernels, r, a, b);
    public MemoryBuffer1D<float, Stride1D.Dense> Add(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Add(r, a, b));

    public void Subtract(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(SubtractKernels, r, a, b);
    public MemoryBuffer1D<float, Stride1D.Dense> Subtract(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Subtract(r, a, b));

    public void ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(ElementwiseMultiplyKernels, r, a, b);
    public MemoryBuffer1D<float, Stride1D.Dense> ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => ElementwiseMultiply(r, a, b));

    public void Divide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(DivideKernels, r, a, b);
    public MemoryBuffer1D<float, Stride1D.Dense> Divide(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Divide(r, a, b));

    public void Max(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(MaxKernels, r, a, b);
    public MemoryBuffer1D<float, Stride1D.Dense> Max(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Max(r, a, b));

    public void Exp(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(ExpKernels, r, val);
    public MemoryBuffer1D<float, Stride1D.Dense> Exp(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Exp(r, val));

    public void Log(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(LogKernels, r, val);
    public MemoryBuffer1D<float, Stride1D.Dense> Log(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Log(r, val));

    public void Sqrt(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(SqrtKernels, r, val);
    public MemoryBuffer1D<float, Stride1D.Dense> Sqrt(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Sqrt(r, val));

    public void Abs(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(AbsKernels, r, val);
    public MemoryBuffer1D<float, Stride1D.Dense> Abs(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Abs(r, val));

    public void Negate(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(NegateKernels, r, val);
    public MemoryBuffer1D<float, Stride1D.Dense> Negate(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Negate(r, val));

    public void ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarPowerKernels, r, value, scalar);
    public MemoryBuffer1D<float, Stride1D.Dense> ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarPower(r, value, scalar));

    public void ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarMultiplyKernels, r, value, scalar);
    public MemoryBuffer1D<float, Stride1D.Dense> ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarMultiply(r, value, scalar));

    public void ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarDivideKernels, r, value, scalar);
    public MemoryBuffer1D<float, Stride1D.Dense> ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarDivide(r, value, scalar));

    public void ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) =>
        Call(ScalarMaxKernels, r, value, scalar);
    public MemoryBuffer1D<float, Stride1D.Dense> ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> value, Value<float> scalar) => 
        Encase(value, r => ScalarMax(r, value, scalar));
    
    public void FloatPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatPowerKernels, r, value, scalar);
    public MemoryBuffer1D<float, Stride1D.Dense> FloatPower(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatPower(r, value, scalar));

    public void FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatMultiplyKernels, r, value, scalar);
    public MemoryBuffer1D<float, Stride1D.Dense> FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatMultiply(r, value, scalar));

    public void FloatMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatMaxKernels, r, value, scalar);
    public MemoryBuffer1D<float, Stride1D.Dense> FloatMax(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatMax(r, value, scalar));

    public MemoryBuffer1D<float, Stride1D.Dense> Sum(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val.AcceleratorIndex(), 1, r => Sum(r, val));
    
    public MemoryBuffer1D<float, Stride1D.Dense> Dot(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Encase(a.AcceleratorIndex(), 1, r => Dot(r, a, b));
}