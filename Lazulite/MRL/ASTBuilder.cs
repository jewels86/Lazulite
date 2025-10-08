using Antlr4.Runtime;

namespace Lazulite.MRL;

public static class ASTBuilder
{
    public static void FromSource(string source)
    {
        AntlrInputStream inputStream = new(source);
        MRLLexer lexer = new(inputStream);
        CommonTokenStream commonTokenStream = new(lexer);
        MRLParser parser = new(commonTokenStream);
        var parseTree = parser.program();
        
    }
    
    
}