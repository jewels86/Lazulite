using Antlr4.Runtime;
using Antlr4.Runtime.Tree;

namespace Lazulite.MRL;

public class MRLVisitor : MRLParserBaseVisitor<IASTNode>
{
    private int depth = 0;
    
    public override IASTNode Visit(IParseTree tree)
    {
        depth++;
        string ident = new string(' ', depth * 2);
        if (tree is ParserRuleContext ctx)
            Console.WriteLine($"{ident} Text: {ctx.GetText()}");
        
        var result = base.Visit(tree);
        depth--;
        return result;
    }
}

public interface IASTNode
{
    public int Line { get; }
    public int Column { get; }
}