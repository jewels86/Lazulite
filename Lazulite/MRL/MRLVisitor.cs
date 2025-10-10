using Antlr4.Runtime;
using Antlr4.Runtime.Tree;

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

public record PrimaryExpression(int Line, int Column, ASTNode Expression) : ASTNode(Line, Column);
 
public enum Modifier
{
    Static, IStatic, Readonly, Constant, IConstant, Dynamic, Required, Nullable, Specific, Same
}