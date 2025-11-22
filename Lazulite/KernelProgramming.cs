using ILGPU;
using ILGPU.Runtime;

namespace Lazulite;

public static class KernelProgramming
{
    public static (double x, double y, double z) GetVector3(ArrayView1D<double, Stride1D.Dense> array, int i) => (array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);
    public static void SetVector3(ArrayView1D<double, Stride1D.Dense> array, int i, (double x, double y, double z) value) => (array[i * 3], array[i * 3 + 1], array[i * 3 + 2]) = value;

    public static (double x, double y, double z) AddVector3((double x, double y, double z) a, (double x, double y, double z) b) => (a.x + b.x, a.y + b.y, a.z + b.z);
    public static (double x, double y, double z) SubtractVector3((double x, double y, double z) a, (double x, double y, double z) b) => (a.x - b.x, a.y - b.y, a.z - b.z);
    public static (double x, double y, double z) MultiplyVector3((double x, double y, double z) a, double b) => (a.x * b, a.y * b, a.z * b);
    public static (double x, double y, double z) DivideVector3((double x, double y, double z) a, double b) => (a.x / b, a.y / b, a.z / b);
    public static double Magnitude2Vector3((double x, double y, double z) a) => a.x * a.x + a.y * a.y + a.z * a.z;
    public static (double x, double y, double z) NegateVector3((double x, double y, double z) a) => (-a.x, -a.y, -a.z);
    
}