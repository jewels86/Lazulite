namespace MRL.Analysis;

public class TypeValidator
{
    public TypeSymbolTable TypeSymbolTable { get; }
    public Dictionary<string, List<string>> TypePassesCache { get; } = [];

    public TypeValidator(TypeSymbolTable typeSymbolTable)
    {
        TypeSymbolTable = typeSymbolTable;
    }

    public bool TypePasses(TypeInfo type, TypeInfo? target)
    {
        if (target is null) return true;
        
        foreach (var targetField in target.Fields.Values)
        {
            if (!type.Fields.TryGetValue(targetField.Name, out FieldInfo? field)) return false;
            if (!FieldPasses(field, targetField, type.Name)) return false;
        }
        foreach (var methods in target.Methods.Values)
        foreach (var targetMethod in methods)
        {
            if (!type.Methods.TryGetValue(targetMethod.Name, out List<MethodInfo>? methodList)) return false;
            if (!methodList.Any(m => MethodPasses(m, targetMethod, type.Name))) return false;
        }
        
        if (TypePassesCache.TryGetValue(type.Name, out var list)) list.Add(target.Name);
        else TypePassesCache[type.Name] = [target.Name];
        return true;
    }

    public bool TypePasses(TypeReference type, TypeReference? target, string typeName)
    {
        if (target is null) return true;
        string targetTypeName = target.Modifiers.HasFlag(TypeModifiers.Same) ? typeName : target.Name;
    
        if (type.Name == targetTypeName) return true;
        if (TypePassesCache.TryGetValue(type.Name, out var passesAs) && passesAs.Contains(targetTypeName)) return true;
    
        if (type.ElementType is not null && target.ElementType is not null)
            return TypePasses(type.ElementType, target.ElementType, typeName);
        
        if (type.ElementType is not null || target.ElementType is not null) 
            return false;
    
        TypeInfo typeInfo = TypeSymbolTable.GetType(type.Name) ?? throw new Exception($"Type {type.Name} not found");
        TypeInfo targetInfo = TypeSymbolTable.GetType(targetTypeName) ?? throw new Exception($"Type {targetTypeName} not found");
    
        return Compatible(type.Modifiers, target.Modifiers) && TypePasses(typeInfo, targetInfo);
    }

    public bool Validate(TypeInfo type)
    {
        if (type.Interfaces.Select(i => TypePasses(type, TypeSymbolTable.GetType(i))).Any()) return false;
        if (type.Alikes.Select(i => TypePasses(type, TypeSymbolTable.GetType(i))).Any()) return false;

        return true;
    }

    #region Members
    public bool FieldPasses(FieldInfo field, FieldInfo target, string typeName)
    {
        if (field.Name != target.Name) return false;
        
        if (!TypePasses(field.Type, target.Type, typeName)) return false;
        if (!Compatible(field.Modifiers, target.Modifiers)) return false;
        
        return true;
    }

    public bool MethodPasses(MethodInfo method, MethodInfo target, string typeName)
    {
        if (method.Name != target.Name) return false;
        if (method.Parameters.Count != target.Parameters.Count) return false;
        foreach (var tuple in method.Parameters.Zip(target.Parameters))
        {
            var (parameter, targetParameter) = tuple;
            if (!TypePasses(parameter.Type, targetParameter.Type, typeName)) return false;
            if (!Compatible(parameter.Type.Modifiers, targetParameter.Type.Modifiers)) return false;
        }
        if (!TypePasses(method.ReturnType, target.ReturnType, typeName)) return false;
        if (!Compatible(method.ReturnType.Modifiers, target.ReturnType.Modifiers)) return false;
        return true;
    }

    #endregion

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