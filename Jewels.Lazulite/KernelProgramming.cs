using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public static class KernelProgramming
{
    public static (float x, float y, float z) GetVector3(ArrayView1D<float, Stride1D.Dense> array, int i) => (array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);
    public static void SetVector3(ArrayView1D<float, Stride1D.Dense> array, int i, (float x, float y, float z) value) => (array[i * 3], array[i * 3 + 1], array[i * 3 + 2]) = value;

    public static (float x, float y, float z) AddVector3((float x, float y, float z) a, (float x, float y, float z) b) => (a.x + b.x, a.y + b.y, a.z + b.z);
    public static (float x, float y, float z) SubtractVector3((float x, float y, float z) a, (float x, float y, float z) b) => (a.x - b.x, a.y - b.y, a.z - b.z);
    public static (float x, float y, float z) MultiplyVector3((float x, float y, float z) a, float b) => (a.x * b, a.y * b, a.z * b);
    public static (float x, float y, float z) DivideVector3((float x, float y, float z) a, float b) => (a.x / b, a.y / b, a.z / b);
    public static float Magnitude2Vector3((float x, float y, float z) a) => a.x * a.x + a.y * a.y + a.z * a.z;
    public static (float x, float y, float z) NegateVector3((float x, float y, float z) a) => (-a.x, -a.y, -a.z);
    
}