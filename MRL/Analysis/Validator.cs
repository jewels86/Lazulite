namespace MRL.Analysis;

public class Validator
{
    public TypeSymbolTable TypeSymbolTable { get; }
    public Dictionary<string, List<string>> TypePassesCache { get; } = [];

    public Validator(TypeSymbolTable typeSymbolTable)
    {
        TypeSymbolTable = typeSymbolTable;
    }
    
    public bool TypeImplements(TypeInfo concreteType, TypeInfo interfaceType)
    {
        
        
    }

    public bool TypePasses(TypeInfo type, TypeInfo target)
    {
        
    }

    public bool TypePasses(TypeReference type, TypeReference target, string typeName)
    {
        string targetTypeName = target.Modifiers.HasFlag(TypeModifiers.Same) ? typeName : target.Name;
    
        if (type.Name == targetTypeName) return true;
        if (TypePassesCache.TryGetValue(type.Name, out var passesAs) && passesAs.Contains(targetTypeName)) return true;
    
        if (type.ElementType is not null && target.ElementType is not null)
            return TypePasses(type.ElementType, target.ElementType, typeName);
        
        if (type.ElementType is not null || target.ElementType is not null) 
            return false;
    
        TypeInfo typeInfo = TypeSymbolTable.GetType(type.Name) ?? throw new Exception($"Type {type.Name} not found");
        TypeInfo targetInfo = TypeSymbolTable.GetType(targetTypeName) ?? throw new Exception($"Type {targetTypeName} not found");
    
        if (!Compatible(type.Modifiers, target.Modifiers)) return false;
    
        return TypePasses(typeInfo, targetInfo);
    }

    public bool Validate(TypeInfo type)
    {
        
    }

    public bool FieldPasses(FieldInfo field, FieldInfo target, string typeName)
    {
        if (field.Name != target.Name) return false;
        
        if (!TypePasses(field.Type, target.Type, typeName)) return false;
        if (!Compatible(field.Modifiers, target.Modifiers)) return false;
        
        return true;
    }

    #region Modifiers
    public static FieldModifiers Transform(FieldModifiers modifiers)
    {
        if (modifiers.HasFlag(FieldModifiers.IStatic) && modifiers.HasFlag(FieldModifiers.Readonly)) modifiers = FieldModifiers.IConstant;
        if (modifiers.HasFlag(FieldModifiers.Static) && modifiers.HasFlag(FieldModifiers.Readonly)) modifiers = FieldModifiers.Constant;
        return modifiers;
    }

    public static bool Compatible(FieldModifiers modifiers, FieldModifiers target)
    {
        modifiers = Transform(modifiers);
        target = Transform(target);
        
        if ((target.HasFlag(FieldModifiers.IConstant) || target.HasFlag(FieldModifiers.Constant)) 
            && !(modifiers.HasFlag(FieldModifiers.IConstant) || modifiers.HasFlag(FieldModifiers.Constant)))
            return false;
        if ((target.HasFlag(FieldModifiers.IStatic) || target.HasFlag(FieldModifiers.Static)) 
            && !(modifiers.HasFlag(FieldModifiers.IStatic) || modifiers.HasFlag(FieldModifiers.Static)))
            return false;
        if (target.HasFlag(FieldModifiers.Readonly) && !modifiers.HasFlag(FieldModifiers.Readonly))
            return false;

        return true;
    }

    public static bool Compatible(TypeModifiers modifiers, TypeModifiers target)
    {
        if (target.HasFlag(TypeModifiers.Nullable) && !modifiers.HasFlag(TypeModifiers.Nullable)) return false;
        if (target.HasFlag(TypeModifiers.Array) && !modifiers.HasFlag(TypeModifiers.Array)) return false;
        if (target.HasFlag(TypeModifiers.Specific) && !modifiers.HasFlag(TypeModifiers.Specific)) return false;
        return true;
    }
    #endregion
}