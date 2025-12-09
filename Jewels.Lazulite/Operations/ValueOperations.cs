namespace Jewels.Lazulite;

public partial class Compute
{
    public Value<T> Add<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Add(a.Data, b.Data));
    public Value<T> Subtract<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Subtract(a.Data, b.Data));
    public Value<T> ElementwiseMultiply<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(ElementwiseMultiply(a.Data, b.Data));
    public Value<T> Divide<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Divide(a.Data, b.Data));
    public Value<T> Max<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Max(a.Data, b.Data));
    
    public Value<T> Exp<T>(Value<T> val) where T : notnull => val.CreateAlike(Exp(val.Data));
    public Value<T> Log<T>(Value<T> val) where T : notnull => val.CreateAlike(Log(val.Data));
    public Value<T> Sqrt<T>(Value<T> val) where T : notnull => val.CreateAlike(Sqrt(val.Data));
    public Value<T> Abs<T>(Value<T> val) where T : notnull => val.CreateAlike(Abs(val.Data));
    public Value<T> Negate<T>(Value<T> val) where T : notnull => val.CreateAlike(Negate(val.Data));
    
    public Value<T> ScalarPower<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarPower(value.Data, scalar));
    public Value<T> ScalarMultiply<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarMultiply(value.Data, scalar));
    public Value<T> ScalarDivide<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarDivide(value.Data, scalar));
    public Value<T> ScalarMax<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarMax(value.Data, scalar));
    
    public Value<T> FloatPower<T>(Value<T> value, float scalar) where T : notnull => value.CreateAlike(FloatPower(value.Data, scalar));
    public Value<T> FloatMultiply<T>(Value<T> value, float scalar) where T : notnull => value.CreateAlike(FloatMultiply(value.Data, scalar));
    public Value<T> FloatMax<T>(Value<T> value, float scalar) where T : notnull => value.CreateAlike(FloatMax(value.Data, scalar));
    
    public Value<float> Sum<T>(Value<T> val) where T : notnull => new ScalarValue(Sum(val.Data));
    public Value<float> Dot<T>(Value<T> a, Value<T> b) where T : notnull => new ScalarValue(Dot(a.Data, b.Data));
}