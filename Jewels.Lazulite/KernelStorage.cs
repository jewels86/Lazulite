namespace Jewels.Lazulite;

public class KernelStorage<T>(T action) where T : notnull
{
    public T Action { get; } = action;
    public T?[] Kernels { get; } = new T?[Compute.Accelerators.Count];
    
    public T? this[int index] { get => Kernels[index]; set => Kernels[index] = value; }
}