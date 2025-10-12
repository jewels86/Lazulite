

using MRL.Parsing;

namespace MRL.Analysis;

public class SymbolTable
{
    public Dictionary<string, TypeInfo> Types { get; set; } = new();

    public static SymbolTable BuildSymbolTable(ProgramNode program)
    {
        SymbolTable symbolTable = new();
        
        foreach (var declaration in program.Declarations.Where(d => d is TypeDeclarationNode).Cast<TypeDeclarationNode>())
        {
            TypeInfo typeInfo = new(
                declaration.Name,
                declaration.Complete,
                declaration.Interfaces.Identifiers.Select(i => i.Name).ToList(),
                declaration.Alikes.Identifiers.Select(i => i.Name).ToList(),
                [], []
            );

            declaration.Members.ForEach(m =>
            {
                if (m is FieldDeclarationNode fieldDeclaration)
                    typeInfo.Fields[fieldDeclaration.Name] = new(
                        fieldDeclaration.Name,
                        CreateTypeReference(fieldDeclaration.Type, fieldDeclaration.modifiers),
                        CreateFieldModifiers(fieldDeclaration.modifiers),
                        fieldDeclaration.Initializer);
                if (m is MethodDeclarationNode methodDeclaration)
                {
                    MethodInfo methodInfo = new(
                        methodDeclaration.MethodSignature.Name,
                        CreateParameterInfoList(methodDeclaration.MethodSignature.DeclaredParameters),
                        CreateTypeReference(methodDeclaration.MethodSignature.ReturnType.Type, methodDeclaration.MethodSignature.ReturnType.modifiers),
                        methodDeclaration.MethodSignature.Inplace,
                        methodDeclaration.MethodSignature.ReturnType.preserves,
                        methodDeclaration.Block);
                    if (!typeInfo.Methods.ContainsKey(methodDeclaration.MethodSignature.Name))
                        typeInfo.Methods[methodDeclaration.MethodSignature.Name] = [];
                    typeInfo.Methods[methodDeclaration.MethodSignature.Name].Add(methodInfo);
                }
                if (m is OperatorDeclarationNode operatorDeclaration)
                {
                    MethodInfo operatorInfo = new(
                        operatorDeclaration.Operator,
                        CreateParameterInfoList(operatorDeclaration.DeclaredParameters),
                        CreateTypeReference(operatorDeclaration.ReturnType.Type, operatorDeclaration.ReturnType.modifiers),
                        operatorDeclaration.Inplace,
                        operatorDeclaration.ReturnType.preserves,
                        operatorDeclaration.Block);
                    if (!typeInfo.Methods.ContainsKey(operatorDeclaration.Operator))
                        typeInfo.Methods[operatorDeclaration.Operator] = [];
                    typeInfo.Methods[operatorDeclaration.Operator].Add(operatorInfo);
                }
            });
            
            symbolTable.Types[declaration.Name] = typeInfo;
        }
        return symbolTable;
    }

    #region Helpers
    private static TypeReference CreateTypeReference(TypeNode typeNode, List<ModifierNode> modifiers)
    {
        int arrayCount = typeNode.ArrayCount;
        if (arrayCount == 0) return new(typeNode.Name, CreateTypeModifiers(modifiers, arrayCount), null);
        
        string elementTypeName = typeNode.Name;
        TypeReference elementType = CreateTypeReference(typeNode with { ArrayCount = 0 }, []);
        return new(elementTypeName, CreateTypeModifiers(modifiers, arrayCount), elementType);
    }

    private static TypeModifiers CreateTypeModifiers(List<ModifierNode> modifiers, int arrayCount)
    {
        TypeModifiers typeModifiers = TypeModifiers.None;
        foreach (var modifier in modifiers)
        {
            switch (modifier.Modifier)
            {
                case Modifier.Nullable: typeModifiers |= TypeModifiers.Nullable; break;
                case Modifier.Specific: typeModifiers |= TypeModifiers.Specific; break;
                case Modifier.Same: typeModifiers |= TypeModifiers.Same; break;
                default: throw new Exception("Unknown modifier");
            }
        }
        if (arrayCount > 0) typeModifiers |= TypeModifiers.Array;
        return typeModifiers;
    }

    private static FieldModifiers CreateFieldModifiers(List<ModifierNode> modifiers)
    {
        FieldModifiers fieldModifiers = FieldModifiers.None;
        foreach (var modifier in modifiers)
        {
            switch (modifier.Modifier)
            {
                case Modifier.Static: fieldModifiers |= FieldModifiers.Static; break;
                case Modifier.IStatic: fieldModifiers |= FieldModifiers.IStatic; break;
                case Modifier.Readonly: fieldModifiers |= FieldModifiers.Readonly; break;
                case Modifier.Constant: fieldModifiers |= FieldModifiers.Constant; break;
                case Modifier.IConstant: fieldModifiers |= FieldModifiers.IConstant; break;
                case Modifier.Dynamic: fieldModifiers |= FieldModifiers.Dynamic; break;
                case Modifier.Required: fieldModifiers |= FieldModifiers.Required; break;
                default: throw new Exception("Unknown modifier");
            }
        }
        return fieldModifiers;
    }

    private static List<ParameterInfo> CreateParameterInfoList(List<DeclaredParameterNode> parameters)
    {
        return parameters
            .Select(p => new ParameterInfo(p.Name, CreateTypeReference(p.Type, p.modifiers)))
            .ToList();
    }
    #endregion
}

#region Records and Enums
public record TypeInfo(
    string Name,
    bool Complete,
    List<string> Interfaces,
    List<string> Alikes,
    Dictionary<string, FieldInfo> Fields,
    Dictionary<string, List<MethodInfo>> Methods
);
        

public record FieldInfo(
    string Name,
    TypeReference Type,
    FieldModifiers Modifiers,
    ASTNode? Initializer
);

public record MethodInfo(
    string Name,
    List<ParameterInfo> Parameters,
    TypeReference ReturnType,
    bool Inplace,
    string? Preserves,
    ASTNode? Body
);

public record ParameterInfo(
    string Name,
    TypeReference Type
);

public record TypeReference(
    string Name,
    TypeModifiers Modifiers,
    TypeReference? ElementType
);

[Flags]
public enum FieldModifiers
{
    None = 0,
    Static = 1,
    IStatic = 2,
    Readonly = 4,
    Constant = 8,
    IConstant = 16,
    Dynamic = 32,
    Required = 64
}

[Flags]
public enum TypeModifiers
{
    None = 0,
    Nullable = 1,
    Specific = 2,
    Same = 4,
    Array = 8
}
#endregion