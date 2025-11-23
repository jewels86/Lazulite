using System.Linq.Expressions;
using ILGPU.Algorithms;

namespace Lazulite;

public static partial class Operations
{
    public static readonly Expression<Func<float, float, float>> AddOp = (a, b) => a + b;
    public static readonly Expression<Func<float, float, float>> SubtractOp = (a, b) => a - b;
    public static readonly Expression<Func<float, float, float>> MultiplyOp = (a, b) => a * b;
    public static readonly Expression<Func<float, float, float>> DivideOp = (a, b) => a / b;
    public static readonly Expression<Func<float, float, float>> ModuloOp = (a, b) => a % b;
    public static readonly Expression<Func<float, float, float>> PowerOp = (a, b) => XMath.Pow(a, b);
}