using ILGPU;
using ILGPU.Runtime;

namespace Jewels.Lazulite;

public partial class kernelProgramming
{
    public static int IndexOf(int i, int stride) => i * stride;
    public static int FromIndex(int index, int stride) => index / stride;
}