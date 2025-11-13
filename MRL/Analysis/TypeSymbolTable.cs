

using MRL.Parsing;

namespace MRL.Analysis;

public class TypeSymbolTable
{
    public Dictionary<string, TypeInfo> Types { get; } = new();
    
    private Dictionary<(string type, string interfaces), bool> _implementsCache = new();
    
    #region Helpers
    public TypeInfo? GetType(string name) => Types.GetValueOrDefault(name);
    public bool TypeExists(string name) => Types.ContainsKey(name);
    
    #endregion

    public static TypeSymbolTable BuildSymbolTable(ProgramNode program)
    {
        TypeSymbolTable typeSymbolTable = new();
        
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
                switch (m)
                {
                    case FieldDeclarationNode fieldDeclaration:
                    {
                        var typeModifiers = fieldDeclaration.modifiers
                            .Where(mod => mod.Modifier is Modifier.Nullable or Modifier.Specific or Modifier.Same)
                            .ToList();
                    
                        typeInfo.Fields[fieldDeclaration.Name] = new(
                            fieldDeclaration.Name,
                            CreateTypeReference(fieldDeclaration.Type, typeModifiers),
                            CreateFieldModifiers(fieldDeclaration.modifiers),
                            fieldDeclaration.Initializer);
                        break;
                    }
                    case MethodDeclarationNode methodDeclaration:
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
                        break;
                    }
                    case OperatorDeclarationNode operatorDeclaration:
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
                        break;
                    }
                }
            });
            
            typeSymbolTable.Types[declaration.Name] = typeInfo;
        }
        return typeSymbolTable;
    }

    #region Static Helpers
    private static TypeReference CreateTypeReference(TypeNode typeNode, List<ModifierNode> modifiers)
    {
        int arrayCount = typeNode.ArrayCount;
        if (arrayCount == 0) return new(typeNode.Name, CreateTypeModifiers(modifiers, arrayCount), null, 0);
        
        string elementTypeName = typeNode.Name;
        TypeReference elementType = CreateTypeReference(typeNode with { ArrayCount = 0 }, []);
        return new(elementTypeName, CreateTypeModifiers(modifiers, arrayCount), elementType, arrayCount);
    }

    private static TypeModifiers CreateTypeModifiers(List<ModifierNode> modifiers, int arrayCount)
    {
        TypeModifiers typeModifiers = TypeModifiers.None;
        foreach (var modifier in modifiers)
        {
            typeModifiers |= modifier.Modifier switch
            {
                Modifier.Nullable => TypeModifiers.Nullable,
                Modifier.Specific => TypeModifiers.Specific,
                Modifier.Same => TypeModifiers.Same,
                _ => throw new Exception("Unknown modifier: " + modifier.Modifier)
            };
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
    TypeReference? ElementType,
    int ArrayCount
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