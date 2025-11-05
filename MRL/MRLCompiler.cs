using MRL.Analysis;
using MRL.Parsing;

namespace MRL;

public static class MRLCompiler
{
    public static void Compile(string source)
    {
        ProgramNode program = (ProgramNode)ASTBuilder.FromSource(source);
        SemanticAnalyzer.Analyze(program);
    }
}