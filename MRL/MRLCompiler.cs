using MRL.Parsing;

namespace MRL;

public static class MRLCompiler
{
    public static void Compile(string source)
    {
        ASTBuilder.FromSource(source);
    }
}