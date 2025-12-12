using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public partial class KernelProgramming
{
    public static int VectorIndexOf(int i, int stride) => i * stride;
    public static int VectorFromIndex(int index, int stride) => index / stride;
}