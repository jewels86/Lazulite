using Antlr4.Runtime;
using Antlr4.Runtime.Tree;

namespace Lazulite.MRL;

public class MRLVisitor : MRLParserBaseVisitor<IASTNode>
{
    public override IASTNode VisitProgram(MRLParser.ProgramContext context)
    {
        var declarations = new List<IASTNode>();
        foreach (var decl in context.declaration())
            declarations.Add(Visit(decl));
        
        return new ProgramNode(context.Start.Line, context.Start.Column, declarations);
    }

    public override IASTNode VisitTypeDeclaration(MRLParser.TypeDeclarationContext context)
    {
        var nameNode = Visit(context.GetChild(0));
        var complete = context.COMPLETE() != null;
        var interfacesNode = Visit(context.GetChild(5));
        var alikesNode = Visit(context.GetChild(7));
        
        string name = nameNode is TypeNode tn ? tn.Name : throw new Exception("Invalid type name node");
        List<IdentifierNode> interfaces = interfacesNode is IdentifierListNode idList ? idList.Identifiers : throw new Exception("Invalid interfaces node");
        List<IdentifierNode> alikes = alikesNode is IdentifierListNode alList ? alList.Identifiers : [];
        List<IASTNode> members = [];
        
        for (int i = 9; i < context.ChildCount - 1; i++)
        {
            var memberNode = Visit(context.GetChild(i));
            members.Add(memberNode);
        }
        
        return new TypeDeclarationNode(context.Start.Line, context.Start.Column, name, complete, 
            interfaces.Select(i => i.Name).ToList(), 
            alikes.Select(a => a.Name).ToList(), members);
    }

    public override IASTNode VisitMethodDeclaration(MRLParser.MethodDeclarationContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        var methodSignatureNode = Visit(context.methodSignature());
        var blockNode = Visit(context.block());
        
        if (methodSignatureNode is not MethodSignatureNode methodSignature)
            throw new Exception("Invalid method signature node");
        if (blockNode is not BlockNode block)
            throw new Exception("Invalid block node");
        
        return new MethodDeclarationNode(line, column, methodSignature, block);
    }

    public override IASTNode VisitVariableDeclaration(MRLParser.VariableDeclarationContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        var typeNode = Visit(context.type());
        var name = context.IDENTIFIER().GetText();
        var modifierNode = Visit(context.modifier(4));
        var expressionNode = context.expression() != null ? Visit(context.expression()) : null;
        
        if (typeNode is not TypeNode type
            || modifierNode is not ModifierNode modifier)
            throw new Exception("Invalid type node or modifier node");

        return new VariableDeclarationNode(line, column, name, type, expressionNode);
    }
}

public interface IASTNode
{
    public int Line { get; }
    public int Column { get; }
}

public record ProgramNode(int Line, int Column, List<IASTNode> Declarations) : IASTNode;
public record TypeDeclarationNode(int Line, int Column, string Name, bool Complete, List<string> Interfaces, List<string> Alikes, List<IASTNode> Members) : IASTNode;
public record MethodDeclarationNode(int Line, int Column, MethodSignatureNode MethodSignature, BlockNode Block) : IASTNode;
public record VariableDeclarationNode(int Line, int Column, string Name, TypeNode Type, IASTNode? expression) : IASTNode;

public record FieldDeclarationNode(int Line, int Column, string Name, TypeNode Type, BlockNode? Initializer) : IASTNode;
public record MethodSignatureNode(int Line, int Column, string Name, List<DeclaredParameterNode> DeclaredParameters, TypeNode ReturnType) : IASTNode;
public record OperatorDeclarationNode(int Line, int Column, string Operator, 
    List<DeclaredParameterNode> DeclaredParameters, TypeNode ReturnType, BlockNode Block) : IASTNode;

public record DeclaredParameterNode(int Line, int Column, string Name, TypeNode Type) : IASTNode;

public record TypeNode(int Line, int Column, string Name, List<string> Modifiers) : IASTNode;

public record BlockNode(int Line, int Column, List<IASTNode> Statements) : IASTNode;

public record LiteralNode(int Line, int Column, string Value) : IASTNode;
public record IdentifierNode(int Line, int Column, string Name) : IASTNode;

public record IdentifierListNode(int Line, int Column, List<IdentifierNode> Identifiers) : IASTNode;
public record ModifierNode(int Line, int Column, List<string> Modifiers) : IASTNode;

