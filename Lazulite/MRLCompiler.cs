using Lazulite.MRL;

namespace Lazulite;

public static class MRLCompiler
{
    public static void Compile(string source)
    {
        ASTBuilder.FromSource(source);
    }
}