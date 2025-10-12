using MRL.Parsing;

namespace MRL.Analysis;

public static class SemanticAnalyzer
{
    public static void Analyze(ProgramNode program)
    {
        SymbolTable symbolTable = SymbolTable.BuildSymbolTable(program);
        
    }
}