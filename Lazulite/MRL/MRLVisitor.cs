using Antlr4.Runtime;
using Antlr4.Runtime.Tree;
using Wasm.Interpret;

namespace Lazulite.MRL;

public class MRLVisitor : MRLParserBaseVisitor<ASTNode>
{
    public override ASTNode VisitProgram(MRLParser.ProgramContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        var declarations = new List<ASTNode>();
        foreach (var decl in context.declaration())
            declarations.Add(Visit(decl));
        
        return new ProgramNode(line, column, declarations);
    }

    public override ASTNode VisitTypeDeclaration(MRLParser.TypeDeclarationContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        string name = context.IDENTIFIER().GetText();
        bool complete = context.COMPLETE() != null;

        IdentifierListNode interfaces = context.interfaces != null
            ? (IdentifierListNode)Visit(context.interfaces)
            : new IdentifierListNode(line, column, []);
        
        IdentifierListNode alikes = context.alikes != null
            ? (IdentifierListNode)Visit(context.alikes)
            : new IdentifierListNode(line, column, []);
        
        var members = context.memberDeclaration()
            .Select(Visit)
            .ToList();

        return new TypeDeclarationNode(line, column, name, complete, interfaces, alikes, members);
    }
    
    #region Statements and Expressions

    public override ASTNode VisitPrimaryExpression(MRLParser.PrimaryExpressionContext context)
    {
        if (context.IDENTIFIER() != null)
        {
            return new IdentifierNode(
                context.Start.Line, 
                context.Start.Column, 
                context.IDENTIFIER().GetText()
            );
        }
    
        return context.literal() != null ? Visit(context.literal()) : Visit(context.expression());
    }

    public override ASTNode VisitCallExpression(MRLParser.CallExpressionContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        if (context.primaryExpression() != null) return Visit(context.primaryExpression());
        if (context.withExpression() != null) return Visit(context.withExpression());

        ASTNode baseExpression = Visit(context.callExpression());
        if (context.DOT() != null) 
            return new MemberAccessNode(context.Start.Line, context.Start.Column, baseExpression, context.IDENTIFIER().GetText());
        
        ParameterListNode args = context.parameterList() != null
            ? (ParameterListNode)Visit(context.parameterList())
            : new(context.Start.Line, context.Start.Column, []);;
        
        return new FunctionCallNode(line, column, baseExpression, args);
    }

    public override ASTNode VisitParameter(MRLParser.ParameterContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        string? name = context.IDENTIFIER()?.GetText();
        ASTNode expression = Visit(context.expression());
        
        return new ParameterNode(line, column, name, expression);
    }

    public override ASTNode VisitUnaryExpression(MRLParser.UnaryExpressionContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Line;

        return context.callExpression() != null 
            ? Visit(context.callExpression()) 
            : new UnaryExpressionNode(line, column, Visit(context.unaryExpression()), context.@operator().GetText());
    }

    public override ASTNode VisitBinaryExpression(MRLParser.BinaryExpressionContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Line;

        if (context.RPAREN().Length != 0)
        {
            List<PartialBinaryExpression> parts = [];
            context.
        }
    }

    #endregion
}

public abstract record ASTNode(int Line, int Column);

public record ProgramNode(int Line, int Column, List<ASTNode> Declarations) : ASTNode(Line, Column);
public record TypeDeclarationNode(int Line, int Column, string Name, bool Complete, 
    IdentifierListNode Interfaces, IdentifierListNode Alikes, List<ASTNode> Members) : ASTNode(Line, Column);
public record MethodDeclarationNode(int Line, int Column, MethodSignatureNode MethodSignature, BlockNode Block) 
    : ASTNode(Line, Column);
public record VariableDeclarationNode(int Line, int Column, string Name, TypeNode Type, ASTNode? expression) 
    : ASTNode(Line, Column);

public record FieldDeclarationNode(int Line, int Column, string Name, TypeNode Type, BlockNode? Initializer) 
    : ASTNode(Line, Column);
public record MethodSignatureNode(int Line, int Column, string Name, List<DeclaredParameterNode> DeclaredParameters, TypeNode ReturnType) 
    : ASTNode(Line, Column);
public record OperatorDeclarationNode(int Line, int Column, string Operator, 
    List<DeclaredParameterNode> DeclaredParameters, TypeNode ReturnType, BlockNode Block) : ASTNode(Line, Column);

public record DeclaredParameterNode(int Line, int Column, string Name, TypeNode Type) : ASTNode(Line, Column);

public record TypeNode(int Line, int Column, string Name, ModifierNode Modifiers) : ASTNode(Line, Column);

public record BlockNode(int Line, int Column, List<ASTNode> Statements) : ASTNode(Line, Column);

public record LiteralNode(int Line, int Column, string Value) : ASTNode(Line, Column);
public record IdentifierNode(int Line, int Column, string Name) : ASTNode(Line, Column);

public record IdentifierListNode(int Line, int Column, List<IdentifierNode> Identifiers) : ASTNode(Line, Column);
public record ModifierNode(int Line, int Column, List<Modifier> Modifiers) : ASTNode(Line, Column);

public record MemberAccessNode(int Line, int Column, ASTNode Base, string Member) : ASTNode(Line, Column);

public record FunctionCallNode(int Line, int Column, ASTNode Base, ParameterListNode Parameters)
    : ASTNode(Line, Column);

public record ParameterNode(int Line, int Column, string? Name, ASTNode expression) : ASTNode(Line, Column);
public record ParameterListNode(int Line, int Column, List<ParameterNode> Parameters) : ASTNode(Line, Column);

public record UnaryExpressionNode(int Line, int Column, ASTNode Operand, string Operator) : ASTNode(Line, Column);

public record PartialBinaryExpression(int Line, int Column, string Operator, ASTNode Right) : ASTNode(Line, Column);
public record BinaryExpressionNode(int Line, int Column, ASTNode Left, PartialBinaryExpression Right) : ASTNode(Line, Column);
 
public enum Modifier
{
    Static, IStatic, Readonly, Constant, IConstant, Dynamic, Required, Nullable, Specific, Same
}