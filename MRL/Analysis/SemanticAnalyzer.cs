using MRL.Parsing;

namespace MRL.Analysis;

public static class SemanticAnalyzer
{
    public static void Analyze(ProgramNode program)
    {
        TypeSymbolTable typeSymbolTable = TypeSymbolTable.BuildSymbolTable(program);
        
    }
}