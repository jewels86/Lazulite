namespace Jewels.Lazulite;

public partial class Compute
{
    public static Value<T> Add<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Add(a.Data, b.Data));
    public static Value<T> Subtract<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Subtract(a.Data, b.Data));
    public static Value<T> ElementwiseMultiply<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(ElementwiseMultiply(a.Data, b.Data));
    public static Value<T> Divide<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Divide(a.Data, b.Data));
    public static Value<T> Max<T>(Value<T> a, Value<T> b) where T : notnull => a.CreateAlike(Max(a.Data, b.Data));
    
    public static Value<T> Exp<T>(Value<T> val) where T : notnull => val.CreateAlike(Exp(val.Data));
    public static Value<T> Log<T>(Value<T> val) where T : notnull => val.CreateAlike(Log(val.Data));
    public static Value<T> Sqrt<T>(Value<T> val) where T : notnull => val.CreateAlike(Sqrt(val.Data));
    public static Value<T> Abs<T>(Value<T> val) where T : notnull => val.CreateAlike(Abs(val.Data));
    public static Value<T> Negate<T>(Value<T> val) where T : notnull => val.CreateAlike(Negate(val.Data));
    public static Value<T> Sine<T>(Value<T> val) where T : notnull => val.CreateAlike(Sine(val.Data));
    public static Value<T> Cosine<T>(Value<T> val) where T : notnull => val.CreateAlike(Cosine(val.Data));
    public static Value<T> Tangent<T>(Value<T> val) where T : notnull => val.CreateAlike(Tangent(val.Data));
    
    public static Value<T> ScalarPower<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarPower(value.Data, scalar));
    public static Value<T> ScalarMultiply<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarMultiply(value.Data, scalar));
    public static Value<T> ScalarDivide<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarDivide(value.Data, scalar));
    public static Value<T> ScalarMax<T>(Value<T> value, Value<float> scalar) where T : notnull => value.CreateAlike(ScalarMax(value.Data, scalar));
    
    public static Value<T> FloatPower<T>(Value<T> value, float scalar) where T : notnull => value.CreateAlike(FloatPower(value.Data, scalar));
    public static Value<T> FloatMultiply<T>(Value<T> value, float scalar) where T : notnull => value.CreateAlike(FloatMultiply(value.Data, scalar));
    public static Value<T> FloatMax<T>(Value<T> value, float scalar) where T : notnull => value.CreateAlike(FloatMax(value.Data, scalar));
    
    public static Value<float> Sum<T>(Value<T> val) where T : notnull => new ScalarValue(Sum(val.Data));
    public static Value<float> Dot<T>(Value<T> a, Value<T> b) where T : notnull => new ScalarValue(Dot(a.Data, b.Data));
}