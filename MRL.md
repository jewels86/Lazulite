# MRL Language Documentation

MRL (Mathematical Representation Language) is a high-level language designed to describe mathematical systems without delving into implementation specifics. It compiles to WASM modules that can be used by any WASM-compatible application.

## Overview

MRL uses an interface-based type system that allows you to describe mathematical concepts and their relationships in a natural, readable way. The language is designed to be flexible enough to represent any system describable with mathematics—from physics simulations to economic models to fluid dynamics.

## Key Features

### Interface-based Typing

MRL uses the `itype` keyword to define types. These are similar to interfaces in other languages, but with more flexibility for mathematical concepts. An `itype` defines a contract that implementations must satisfy.

```
vector_t: itype tensor_t = {
    ndims: iconstant int = 1,
    length: int,
    magnitude: scalar_t,
    normalized: vector_t,
    dot(other: same vector_t) -> scalar_t,
}
```

### Type Modifiers

- **`static`**: Defines values that belong to the type itself rather than implementations or instances of the type.
- **`istatic`**: Defines static members that belong to the implementation of the type.
- **`readonly`**: Indicates that a member cannot be modified after its initial assignment.
- **`constant`/`iconstant`**: Equivalent to `static readonly` or `istatic readonly`, used for defining constants.
- **`specific`**: Defines members that are specific to a particular implementation of a type and not its inheritors.
- **`dynamic`**: Indicates that a member's value will be determined at runtime, allowing for computed properties.
- **`required`**: Indicates that a member must be provided when creating an instance.
- **`nullable`**: Indicates that a member can have a null value.
- **`inplace`**: Indicates that an operator or method modifies the receiver rather than creating a new instance.

### Operators

The `operator` keyword is used to define custom operators for types. The `this` keyword explicitly indicates the receiver of the operation:

```
operator this +(other: vector3) -> vector3 = {
    return this with {
        x = this.x + other.x,
        y = this.y + other.y,
        z = this.z + other.z,
    };
}

operator this *(scalar: scalar_t) -> vector3 = {
    return this with {
        x = this.x * scalar,
        y = this.y * scalar,
        z = this.z * scalar,
    };
}
```

For in-place operations, use the `inplace` keyword:

```
operator this +=(other: vector3) inplace = {
    this.x := this.x + other.x;
    this.y := this.y + other.y;
    this.z := this.z + other.z;
}
```

### Constructors

The `new` keyword combined with `operator` is used to define constructor operations:

```
operator new this(x: scalar_t, y: scalar_t, z: scalar_t) -> vector3 = {
    this.x = x;
    this.y = y;
    this.z = z;
}
```

### The `alike` Keyword

When declaring a type, `alike` indicates that the type can substitute for another type without strict inheritance:

```
vector4: complete itype vector_t alike vector3 {
    # vector4 can be used anywhere vector3 is expected
}
```

A type that is `alike` another type:
- Can be passed anywhere the target type is expected
- Inherits methods from the target type by default (unless overridden)
- Must satisfy the target type's interface
- May have additional members that the target type doesn't have

This is useful when you want type compatibility without forcing strict inheritance. For example, `vector4` has all the properties of `vector3` (x, y, z) plus an additional `w` component. It can be used anywhere `vector3` is expected, but maintains its own type identity.

**Important**: When a type inherits operators through `alike`, operations preserve hidden fields from the left operand (`this`) but not the right operand. If you need to preserve both operands, provide a specific overload:

```
# Inherited from vector3 - preserves this.w but not other.w
operator this +(other: vector3) -> vector3 = {
    return this with { x = this.x + other.x, y = this.y + other.y, z = this.z + other.z };
}

# Explicit overload for vector4 + vector4
operator this +(other: vector4) -> vector4 = {
    return this with {
        x = this.x + other.x,
        y = this.y + other.y,
        z = this.z + other.z,
        w = this.w + other.w,
    };
}
```

### The `same` Keyword

In method signatures, `same` indicates "the same type as the implementation":

```
dot(other: same vector_t) -> scalar_t
```

When implemented in `vector3`, this becomes `dot(other: vector3) -> scalar_t`. When implemented in `vector4`, it becomes `dot(other: vector4) -> scalar_t`.

### Type Reflection

MRL provides runtime type information through `typeof`:

```
typeof(a).new(...)  # Create new instance of the same type as 'a'
typeof(a).clone()   # Clone an instance maintaining its type
```

This is particularly useful in generic code where you need to create objects without knowing their concrete type at compile time. It ensures type consistency when working with interface types.

**Best Practice**: When writing code that operates on interface types, use `typeof` to create new instances rather than hardcoding concrete types. This prevents accidentally creating the wrong type when working with implementations you don't know about.

### Complete Types

The `complete` keyword indicates that a type is a full implementation of an interface and can be instantiated:

```
vector3: complete itype vector_t = {
    # Full implementation here
}
```

## Preservation and Modification

MRL provides several mechanisms to ensure data integrity when working with types that may have hidden fields or when modifying objects.

### The `with` Keyword

The `with` keyword creates a new instance of a type based on an existing instance, modifying only specified members. **Crucially, `with` preserves all hidden fields from the source object**, including those from subtype implementations:

```
new_vector = existing_vector with { x = 10 }
```

If `existing_vector` is actually a `vector4` (with a hidden `w` field) being used as a `vector3`, the resulting `new_vector` will also be a `vector4` with the original `w` value preserved.

Example in an operator:

```
vector3: complete itype vector_t = {
    x: scalar_t,
    y: scalar_t,
    z: scalar_t,
    
    operator this +(other: vector3) -> vector3 = {
        return this with {
            x = this.x + other.x,
            y = this.y + other.y,
            z = this.z + other.z,
        };
    }
}

v4: vector4 = new vector4(1, 2, 3, 4);
v3: vector3 = new vector3(5, 6, 7);
result: vector3 = v4 + v3;  # result is actually a vector4 with w = 4 preserved
```

### The `out` Keyword

The `out` keyword indicates that a parameter should be modified in place rather than replaced. This preserves all hidden members of any type that satisfies the interface:

```
transform(out position: vector3, out velocity: vector3, time: scalar_t) = {
    position := position + (velocity * time);  # Modifies position, preserves hidden fields
}
```

**Compiler Enforcement**: If a parameter is marked `out`, the compiler will error if you attempt to reassign it with `=`:

```
transform(out position: vector3, ...) = {
    position = new vector3(...);  # ERROR: Cannot replace out parameter
    position := new vector3(...);  # OK: Modifies position in place
}
```

### The `preserves` Keyword

The `preserves` keyword indicates that a function returns the same instance that was passed in, allowing for method chaining:

```
normalize(vec: vector3) -> preserves vec = {
    let magnitude = vec.magnitude;
    vec := vec / magnitude;
    return vec;
}
```

This allows code like:

```
result = normalize(my_vector).scale(2.0);
```

Where each operation modifies and returns the same instance.

### Assignment vs Modification Operators

MRL distinguishes between assignment and modification:

**Assignment (`=`)**: Replaces a reference with a new object
```
x = new vector3(1, 2, 3);  # x now points to a completely new vector3
```

**Modification (`:=`)**: Updates an existing object in place
```
x := new vector3(1, 2, 3);  # Modifies x's fields to match the new values
```

The modification operator is required in `out` and `preserves` contexts, and is generally preferred even when not required, as it preserves hidden fields.

**Best Practice**: Use `:=` by default unless you specifically need to replace an object reference. This preserves hidden type information and is more efficient.

### Type Casting

MRL supports explicit type conversion through casting operators:

**Cast from other types to this type:**
```
scientific_number: complete itype scalar_t = {
    mantissa: float,
    exponent: int,
    
    operator from int(value: int) -> scientific_number = {
        return new scientific_number(value as float, 0);
    }
    
    operator from float(value: float) -> scientific_number = {
        # Normalize to scientific notation
        let exp = floor(log10(abs(value)));
        let mant = value / pow(10, exp);
        return new scientific_number(mant, exp);
    }
}
```

**Cast from this type to other types:**
```
operator to int() -> int = {
    return (this.mantissa * pow(10, this.exponent)) as int;
}

operator to float() -> float = {
    return this.mantissa * pow(10, this.exponent);
}
```

Casting is automatically invoked when types need to be converted:

```
vector3: complete itype vector_t = {
    x: scientific_number,
    y: scientific_number,
    z: scientific_number,
}

v: vector3 = new vector3(1, 2, 3);  # Integers are cast to scientific_number
```

### Custom Modification Operators

Types can define how they should be modified when used with `:=`:

```
scientific_number: complete itype scalar_t = {
    mantissa: float,
    exponent: int,
    
    operator this :=(value: int) inplace = {
        this.mantissa := value as float;
        this.exponent := 0;
    }
    
    operator this :=(value: float) inplace = {
        let exp = floor(log10(abs(value)));
        this.mantissa := value / pow(10, exp);
        this.exponent := exp;
    }
}
```

This allows complex types to handle modification appropriately while maintaining their invariants.

## Design Philosophy

MRL prioritizes:

1. **Readability**: Syntax that closely resembles natural language and mathematical notation
2. **Flexibility**: Interface-based design allows for multiple implementations of the same concept
3. **Type Safety**: Strong typing with runtime type information when needed
4. **Data Integrity**: Preservation mechanisms prevent accidental loss of hidden type information
5. **Modularity**: Separation of concerns (e.g., physics backend vs integration backend)
6. **Reusability**: Code written for interfaces works with any valid implementation

## Compilation

MRL compiles to WASM modules, making the resulting code portable and usable in any WASM-compatible environment. The compiler resolves interface types, generates efficient code, and handles dynamic property computation at compile time where possible.

The compiler also performs preservation analysis to verify that:
- `out` parameters are not reassigned
- `preserves` functions return the correct instance
- Type conversions are valid and preserve necessary information

## Example Use Cases

- Physics simulations (classical mechanics, relativity, quantum systems)
- Mathematical modeling (differential equations, optimization problems)
- Data transformations (tensor operations, signal processing)
- Economic simulations (agent-based models, game theory)
- Any domain that can be expressed mathematically