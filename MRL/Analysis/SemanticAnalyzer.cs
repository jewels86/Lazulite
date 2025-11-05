using MRL.Parsing;

namespace MRL.Analysis;

public static class SemanticAnalyzer
{
    public static void Analyze(ProgramNode program)
    {
        TypeSymbolTable typeSymbolTable = TypeSymbolTable.BuildSymbolTable(program);
        TypeValidator typeValidator = new(typeSymbolTable);
        foreach (TypeInfo type in typeSymbolTable.Types.Values.Where(type => !typeValidator.Validate(type)))
            throw new Exception($"Type {type.Name} is not valid");
    }
}