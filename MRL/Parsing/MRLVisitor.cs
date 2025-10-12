using Antlr4.Runtime;
using Antlr4.Runtime.Tree;

namespace MRL.Parsing;

public class MRLVisitor : MRLParserBaseVisitor<ASTNode>
{
    public override ASTNode VisitProgram(MRLParser.ProgramContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        var declarations = context.declaration()
            .Select(Visit)
            .ToList();

        return new ProgramNode(line, column, declarations);
    }

    #region Declarations
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

    public override ASTNode VisitVariableDeclaration(MRLParser.VariableDeclarationContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;

        string name = context.IDENTIFIER().GetText();
        TypeNode? type = context.type() is not null ? (TypeNode?)Visit(context.type()) : null;
        List<ModifierNode> modifiers = context.modifier().Length != 0
            ? context.modifier().Select(Visit).Cast<ModifierNode>().ToList()
            : [];
        ASTNode? expression = context.expression() != null ? Visit(context.expression()) : null;
        
        return new VariableDeclarationNode(line, column, name, type, modifiers, expression);
    }

    public override ASTNode VisitMethodDeclaration(MRLParser.MethodDeclarationContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        MethodSignatureNode methodSignature = (MethodSignatureNode)Visit(context.methodSignature());
        BlockNode block = (BlockNode)Visit(context.block());
        
        return new MethodDeclarationNode(line, column, methodSignature, block);
    }

    public override ASTNode VisitMemberDeclaration(MRLParser.MemberDeclarationContext context)
    {
        if (context.fieldDeclaration() != null) return Visit(context.fieldDeclaration());
        if (context.methodSignature() != null) return Visit(context.methodSignature());
        return context.methodDeclaration() != null ? Visit(context.methodDeclaration()) : Visit(context.operatorDeclaration());
    }

    public override ASTNode VisitFieldDeclaration(MRLParser.FieldDeclarationContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        string name = context.IDENTIFIER().GetText();
        TypeNode type = (TypeNode)Visit(context.type());
        List<ModifierNode> modifiers = context.modifier().Length != 0
            ? context.modifier().Select(Visit).Cast<ModifierNode>().ToList()
            : [];
        BlockNode? initializer = context.initializer() != null ? (BlockNode)Visit(context.initializer()) : null;
        
        return new FieldDeclarationNode(line, column, name, type, modifiers, initializer);
    }

    public override ASTNode VisitOperatorDeclaration(MRLParser.OperatorDeclarationContext context)
    {
        if (context.NEW() is null)
        {
            string thisName = context.IDENTIFIER()[0].GetText();
            string op = context.declarableOperator().GetText();
            List<DeclaredParameterNode> parameters = ((DeclaredParameterListNode)Visit(context.declaredParameterList())).DeclaredParameters;
            bool inplace = context.INPLACE() != null;
            string? preserves = context.PRESERVES() != null ? context.IDENTIFIER()[1].GetText() : null;
            string? returnType = context.type() != null ? ((TypeNode)Visit(context.type())).Name : null;
            BlockNode block = (BlockNode)Visit(context.block());

            return new OperatorDeclarationNode(
                context.Start.Line,
                context.Start.Column,
                op,
                parameters,
                new ReturnTypeNode(context.Start.Line, context.Start.Column, returnType != null
                    ? new TypeNode(context.Start.Line, context.Start.Column, returnType)
                    : null, preserves),
                block
            );
        }
        else
        {
            string parentType = GetParentType(context)!.IDENTIFIER().GetText();
            List<DeclaredParameterNode> parameters = ((DeclaredParameterListNode)Visit(context.declaredParameterList())).DeclaredParameters;
            BlockNode block = (BlockNode)Visit(context.block());
            parameters.Add(new DeclaredParameterNode(context.Start.Line, context.Start.Column, "this", [], new TypeNode(context.Start.Line, context.Start.Column, parentType!)));
            
            return new OperatorDeclarationNode(
                context.Start.Line,
                context.Start.Column,
                "new",
                parameters,
                new ReturnTypeNode(context.Start.Line, context.Start.Column, parentType != null
                    ? new TypeNode(context.Start.Line, context.Start.Column, parentType)
                    : null, null),
                block
            );
        }
    }

    public override ASTNode VisitDeclaration(MRLParser.DeclarationContext context)
    {
        if (context.typeDeclaration() is not null) return Visit(context.typeDeclaration());
        return context.methodDeclaration() is not null
            ? Visit(context.methodDeclaration())
            : Visit(context.variableDeclaration());
    }

    #endregion
    
    #region Statements and Expressions
    #region Expressions
    public override ASTNode VisitExpression(MRLParser.ExpressionContext context)
    {
        return Visit(context.logicalOrExpression());
    }
    
    public override ASTNode VisitLogicalOrExpression(MRLParser.LogicalOrExpressionContext context)
    {
        var expressions = context.logicalAndExpression()
            .Select(Visit)
            .ToList();
    
        return BuildBinaryExpressionChain(context, expressions, "||");
    }
    
    public override ASTNode VisitLogicalAndExpression(MRLParser.LogicalAndExpressionContext context)
    {
        var expressions = context.equalityExpression()
            .Select(Visit)
            .ToList();
    
        return BuildBinaryExpressionChain(context, expressions, "&&");
    }
    
    public override ASTNode VisitEqualityExpression(MRLParser.EqualityExpressionContext context)
    {
        var expressions = context.relationalExpression()
            .Select(Visit)
            .ToList();
    
        return BuildBinaryExpressionChain(context, expressions);
    }
    
    public override ASTNode VisitRelationalExpression(MRLParser.RelationalExpressionContext context)
    {
        var expressions = context.additiveExpression()
            .Select(Visit)
            .ToList();
        
        return BuildBinaryExpressionChain(context, expressions);
    }
    
    public override ASTNode VisitAdditiveExpression(MRLParser.AdditiveExpressionContext context)
    {
        var expressions = context.multiplicativeExpression()
            .Select(Visit)
            .ToList();
    
        return BuildBinaryExpressionChain(context, expressions, "+");
    }
    
    public override ASTNode VisitMultiplicativeExpression(MRLParser.MultiplicativeExpressionContext context)
    {
        // (expr)(expr)
        if (context.LPAREN().Length > 0)
        {
            var expressions = context.exponentiationExpression()
                .Select(Visit)
                .ToList();

            return BuildBinaryExpressionChain(context, expressions, "*");
        }
        
        var exprs = context.exponentiationExpression()
            .Select(Visit)
            .ToList();
    
        return BuildBinaryExpressionChain(context, exprs);
    }
    
    public override ASTNode VisitExponentiationExpression(MRLParser.ExponentiationExpressionContext context)
    {
        var expressions = context.unaryExpression()
            .Select(Visit)
            .ToList();
    
        return BuildBinaryExpressionChain(context, expressions, "^", true);
    }
    
    public override ASTNode VisitUnaryExpression(MRLParser.UnaryExpressionContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;

        if (context.callExpression() != null)
            return Visit(context.callExpression());
    
        string op = context.children[0].GetText();
        return new UnaryExpressionNode(
            line,
            column,
            Visit(context.unaryExpression()),
            op
        );
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

    public override ASTNode VisitWithExpression(MRLParser.WithExpressionContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
    
        ASTNode baseExpr = Visit(context.primaryExpression());
    
        List<FieldAssignmentNode> updates = [];
        var identifiers = context.IDENTIFIER();
        var expressions = context.expression();

        updates.AddRange(identifiers
            .Select((t, i) => new FieldAssignmentNode(t.Symbol.Line, t.Symbol.Column, t.GetText(), Visit(expressions[i]))));

        return new WithExpressionNode(line, column, baseExpr, updates);
    }

    public override ASTNode VisitLambdaExpression(MRLParser.LambdaExpressionContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;

        List<ParameterNode> parameters = context.parameterList() != null
            ? ((ParameterListNode)Visit(context.parameterList())).Parameters
            : [];
    
        BlockNode body = (BlockNode)Visit(context.block());
    
        return new LambdaExpressionNode(line, column, parameters, body);
    }

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
    #region Statements

    public override ASTNode VisitStatement(MRLParser.StatementContext context)
    {
        if (context.partialStatement() != null) return Visit(context.partialStatement());
        return context.ifStatement() != null ? Visit(context.ifStatement()) : Visit(context.foreachStatement());
    }

    public override ASTNode VisitPartialStatement(MRLParser.PartialStatementContext context)
    {
        if (context.RETURN() is not null)
            return new ReturnStatementNode(context.Start.Line, context.Start.Column, Visit(context.expression()));
        return context.variableDeclaration() is not null 
            ? Visit(context.variableDeclaration()) 
            : new AssignmentStatementNode(
                context.Start.Line, context.Start.Column, 
                Visit(context.callExpression()), 
                context.assignmentOperator().GetText(), 
                Visit(context.expression()));
    }

    public override ASTNode VisitIfStatement(MRLParser.IfStatementContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
    
        ASTNode condition = Visit(context.logicalAndExpression(0));
        BlockNode thenBlock = (BlockNode)Visit(context.block(0));
    
        ASTNode? elseBlock = null;

        if (context.ELSE() == null) return new IfStatementNode(line, column, condition, thenBlock, elseBlock);
        if (context.logicalAndExpression().Length > 1)
        {
            ASTNode elseIfCondition = Visit(context.logicalAndExpression(1));
            BlockNode elseIfBlock = (BlockNode)Visit(context.block(1));
            
            elseBlock = new IfStatementNode(
                context.logicalAndExpression(1).Start.Line,
                context.logicalAndExpression(1).Start.Column,
                elseIfCondition,
                elseIfBlock,
                null
            );
        }
        else
            elseBlock = (BlockNode)Visit(context.block(1));

        return new IfStatementNode(line, column, condition, thenBlock, elseBlock);
    }

    public override ASTNode VisitForeachStatement(MRLParser.ForeachStatementContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;

        string variable = context.IDENTIFIER().GetText();
        ASTNode collection = Visit(context.expression());
        LambdaExpressionNode? lambda = (LambdaExpressionNode?)Visit(context.lambdaExpression());
        BlockNode block = (BlockNode)Visit(context.block());
        
        return new ForeachStatementNode(line, column, collection, variable, lambda, block);
    }
    #endregion
    #endregion
    
    #region Small Stuff
    public override ASTNode VisitParameter(MRLParser.ParameterContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        string? name = context.IDENTIFIER()?.GetText();
        ASTNode expression = Visit(context.expression());
        
        return new ParameterNode(line, column, name, expression);
    }
    
    public override ASTNode VisitLiteral(MRLParser.LiteralContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        string value = context.children[0].GetText();
        return new LiteralNode(line, column, value);
    }

    public override ASTNode VisitBlock(MRLParser.BlockContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        List<ASTNode> statements = context.statement()
            .Select(Visit)
            .ToList();
        
        return new BlockNode(line, column, statements);
    }

    public override ASTNode VisitDeclaredParameter(MRLParser.DeclaredParameterContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        string name = context.IDENTIFIER().GetText();
        List<ModifierNode> modifiers = context.modifier().Length != 0
            ? context.modifier().Select(Visit).Cast<ModifierNode>().ToList()
            : [];
        TypeNode type = (TypeNode)Visit(context.type());
        
        return new DeclaredParameterNode(line, column, name, modifiers, type);
    }

    public override ASTNode VisitDeclaredParameterList(MRLParser.DeclaredParameterListContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        List<DeclaredParameterNode> parameters = context.nextDeclaredParameter()
            .Select(Visit)
            .Cast<DeclaredParameterNode>()
            .ToList();
        if (context.declaredParameter() is not null) parameters.Insert(0, (DeclaredParameterNode)Visit(context.declaredParameter()));
        return new DeclaredParameterListNode(line, column, parameters);
    }

    public override ASTNode VisitIdentifierList(MRLParser.IdentifierListContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        List<IdentifierNode> identifiers = context.IDENTIFIER()
            .Select(t => new IdentifierNode(t.Symbol.Line, t.Symbol.Column, t.GetText()))
            .ToList();
        return new IdentifierListNode(line, column, identifiers);
    }

    public override ASTNode VisitInitializer(MRLParser.InitializerContext context)
    {
        if (context.expression() is not null) return Visit(context.expression());
        if (context.GET() is not null)
        {
            return new DynamicFieldInitializeNode(context.Start.Line, context.Start.Column, 
                (BlockNode)Visit(context.block(0)), 
                context.block().Length < 2 ? null : (BlockNode)Visit(context.block(1)));
        }
        return Visit(context.block(0));
    }

    public override ASTNode VisitMethodSignature(MRLParser.MethodSignatureContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;

        string name = context.IDENTIFIER(0).GetText();
        List<DeclaredParameterNode> parameters = ((DeclaredParameterListNode)Visit(context.declaredParameterList())).DeclaredParameters;
        if (GetParentType(context) is { } parentTypeDeclarationContext)
        {
            string parentType = parentTypeDeclarationContext.IDENTIFIER().GetText();
            parameters.Add(new DeclaredParameterNode(context.Start.Line, context.Start.Column, "this", [], new TypeNode(context.Start.Line, context.Start.Column, parentType!)));
        }
        
        bool inplace = context.INPLACE() != null;
        string? preserves = context.PRESERVES() != null ? context.IDENTIFIER()[1].GetText() : null;
        string? returnType = context.type() != null ? ((TypeNode)Visit(context.type())).Name : null;
        
        return new MethodSignatureNode(line, column, name, parameters, preserves != null 
            ? new ReturnTypeNode(line, column, null, preserves) 
            : new ReturnTypeNode(line, column, returnType != null 
                ? new TypeNode(line, column, returnType) 
                : null, null), inplace);
    }

    public override ASTNode VisitType(MRLParser.TypeContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        string name = context.IDENTIFIER().GetText();
        int arrayCount = context.LBRACK().Length;
        return new TypeNode(line, column, name, arrayCount);
    }

    public override ASTNode VisitParameterList(MRLParser.ParameterListContext context)
    {
        int line = context.Start.Line;
        int column = context.Start.Column;
        
        List<ParameterNode> parameters = context.parameter()
            .Select(Visit)
            .Cast<ParameterNode>()
            .ToList();
        return new ParameterListNode(line, column, parameters);
    }

    #endregion
    
    #region Helper Methods
    private static ASTNode BuildBinaryExpressionChain(IParseTree context, List<ASTNode> expressions, string operatorText, bool rightAssociative = false)
    {
        if (expressions.Count == 1)
            return expressions[0];
    
        int line = ((ParserRuleContext)context).Start.Line;
        int column = ((ParserRuleContext)context).Start.Column;
    
        if (rightAssociative)
        {
            ASTNode result = expressions[^1];
            for (int i = expressions.Count - 2; i >= 0; i--)
            {
                result = new BinaryExpressionNode(line, column, expressions[i], operatorText, result);
            }
            return result;
        }
        else
        {
            ASTNode result = expressions[0];
            for (int i = 1; i < expressions.Count; i++)
            {
                result = new BinaryExpressionNode(line, column, result, operatorText, expressions[i]);
            }
            return result;
        }
    }
    
    private static ASTNode BuildBinaryExpressionChain(ParserRuleContext context, List<ASTNode> expressions)
    {
        if (expressions.Count == 1)
            return expressions[0];
    
        ASTNode result = expressions[0];
        for (int i = 0; i < expressions.Count - 1; i++)
        {
            string op = context.children[i * 2 + 1].GetText();
            result = new BinaryExpressionNode(
                context.Start.Line,
                context.Start.Column,
                result,
                op,
                expressions[i + 1]
            );
        }
        return result;
    }
    
    private MRLParser.TypeDeclarationContext? GetParentType(RuleContext context)
    {
        var parent = context.Parent;
        while (parent != null && parent is not MRLParser.TypeDeclarationContext)
        {
            parent = parent.Parent;
        }
        return parent as MRLParser.TypeDeclarationContext;
    }
    #endregion
}

public abstract record ASTNode(int Line, int Column);

public record ProgramNode(int Line, int Column, List<ASTNode> Declarations) : ASTNode(Line, Column);
public record TypeDeclarationNode(int Line, int Column, string Name, bool Complete, 
    IdentifierListNode Interfaces, IdentifierListNode Alikes, List<ASTNode> Members) : ASTNode(Line, Column);
public record MethodDeclarationNode(int Line, int Column, MethodSignatureNode MethodSignature, BlockNode Block) 
    : ASTNode(Line, Column);
public record VariableDeclarationNode(int Line, int Column, string Name, TypeNode? Type, List<ModifierNode> Modifiers, ASTNode? Expression) 
    : ASTNode(Line, Column);

public record FieldDeclarationNode(int Line, int Column, string Name, TypeNode Type, List<ModifierNode> modifiers, ASTNode? Initializer) 
    : ASTNode(Line, Column);
public record MethodSignatureNode(int Line, int Column, string Name, List<DeclaredParameterNode> DeclaredParameters, ReturnTypeNode ReturnType, bool Inplace = false)   
    : ASTNode(Line, Column);
public record OperatorDeclarationNode(int Line, int Column, string Operator, 
    List<DeclaredParameterNode> DeclaredParameters, ReturnTypeNode ReturnType, BlockNode Block) : ASTNode(Line, Column);

public record DeclaredParameterNode(int Line, int Column, string Name, List<ModifierNode> modifiers, TypeNode Type) : ASTNode(Line, Column);
public record DeclaredParameterListNode(int Line, int Column, List<DeclaredParameterNode> DeclaredParameters) : ASTNode(Line, Column);

public record TypeNode(int Line, int Column, string Name, int ArrayCount = 0) : ASTNode(Line, Column);
public record ReturnTypeNode(int Line, int Column, TypeNode? Type, string? preserves) : ASTNode(Line, Column);

public record BlockNode(int Line, int Column, List<ASTNode> Statements) : ASTNode(Line, Column);

public record LiteralNode(int Line, int Column, string Value) : ASTNode(Line, Column);
public record IdentifierNode(int Line, int Column, string Name) : ASTNode(Line, Column);

public record IdentifierListNode(int Line, int Column, List<IdentifierNode> Identifiers) : ASTNode(Line, Column);
public record ModifierNode(int Line, int Column, Modifier Modifier) : ASTNode(Line, Column);

public record MemberAccessNode(int Line, int Column, ASTNode Base, string Member) : ASTNode(Line, Column);

public record FunctionCallNode(int Line, int Column, ASTNode Base, ParameterListNode Parameters)
    : ASTNode(Line, Column);

public record ParameterNode(int Line, int Column, string? Name, ASTNode expression) : ASTNode(Line, Column);
public record ParameterListNode(int Line, int Column, List<ParameterNode> Parameters) : ASTNode(Line, Column);

public record UnaryExpressionNode(int Line, int Column, ASTNode Operand, string Operator) : ASTNode(Line, Column);

public record BinaryExpressionNode(int Line, int Column, ASTNode Left, string Operator, ASTNode Right) 
    : ASTNode(Line, Column);

public record WithExpressionNode(int Line, int Column, ASTNode Base, List<FieldAssignmentNode> Updates) 
    : ASTNode(Line, Column);

public record FieldAssignmentNode(int Line, int Column, string FieldName, ASTNode Value) 
    : ASTNode(Line, Column);

public record LambdaExpressionNode(int Line, int Column, List<ParameterNode> Parameters, BlockNode Body) 
    : ASTNode(Line, Column);

public record ReturnStatementNode(int Line, int Column, ASTNode? Expression) : ASTNode(Line, Column);
public record AssignmentStatementNode(int Line, int Column, ASTNode Left, string op, ASTNode Right) : ASTNode(Line, Column);
public record IfStatementNode(int Line, int Column, ASTNode Condition, BlockNode ThenBlock, ASTNode? ElseBlock) 
    : ASTNode(Line, Column);
public record ForeachStatementNode(int Line, int Column, ASTNode Collection, string Variable, LambdaExpressionNode? condition, BlockNode Block) 
    : ASTNode(Line, Column);

public record DynamicFieldInitializeNode(int Line, int Column, BlockNode Get, BlockNode? Set) : ASTNode(Line, Column);

public enum Modifier
{
    Static, IStatic, Readonly, Constant, IConstant, Dynamic, Required, Nullable, Specific, Same
}
