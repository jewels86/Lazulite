# MRL Language Documentation
MRL (Mathematical Representation Language) is a high-level language designed to describe mathematical systems without delving into implementation specifics.
It compiles to WASM modules that can be used by any WASM-compatible application.

## Overview

MRL uses an interface-based type system that allows you to describe mathematical concepts and their relationships in a natural, readable way.
The language is designed to be flexible enough to represent any system describable with mathematics—from physics simulations to economic models to fluid dynamics.

## Key Features

### Interface-based Typing
MRL uses the `itype` keyword to define types. 
These are similar to interfaces in other languages, but with more flexibility for mathematical concepts. 
An `itype` defines a contract that implementations must satisfy.

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

### Operators
The `operator` keyword is used to define custom operators for types, enhancing expressiveness:

```
operator +(other: vector3) -> vector3 = { return new vector3(x + other.x, y + other.y, z + other.z); }
operator *(scalar: scalar_t) -> vector3 = { return new vector3(x * scalar, y * scalar, z * scalar); }
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

### The `with`, `out`, and `preserves` Keywords
`with` is used to create a new instance of a type based on an existing instance, modifying only specified members:

```
new_vector = existing_vector with { x = 10; }
```

`out` is used to indicate that a parameter should be changed but not replaced; thus, any type that satisfies the interface can be passed and will have it's hidden members preserved:

```
vector3: complete itype vector_t = {
    x: scalar_t,
    y: scalar_t,
    z: scalar_t,
    
    # Implementation...
    
    operator +(other: vector3) -> vector3 = {
        return this with {
            x = x + other.x,
            y = y + other.y,
            z = z + other.z,
        };
    }
}

transform(out position: vector3, out velocity: vector3, time: scalar_t) = {
    position += velocity * time;
}
```

`preserves` functions like `out` and indicates that a function returns the same instance that was passed in, allowing for method chaining:

```
normalize(vec: vector3) -> preserves vec = {
    magnitude = vec.magnitude;
    vec /= magnitude;
    return vec;
}
```

Compiler warnings will be generated if:
- The instance passed to an `out` parameter is replaced rather than modified: `position = new vector3(...)` is not allowed.
- An operation on a `preserves` return value results in a new instance rather than modifying the existing one: `normalize(vec) + another_vec` is not allowed unless the `+` operator uses `out` or `preserves`.

## Design Philosophy

MRL prioritizes:

1. **Readability**: Syntax that closely resembles natural language and mathematical notation
2. **Flexibility**: Interface-based design allows for multiple implementations of the same concept
3. **Type Safety**: Strong typing with runtime type information when needed
4. **Modularity**: Separation of concerns (e.g., physics backend vs integration backend)
5. **Reusability**: Code written for interfaces works with any valid implementation

## Compilation

MRL compiles to WASM modules, making the resulting code portable and usable in any WASM-compatible environment. The compiler resolves interface types, generates efficient code, and handles dynamic property computation at compile time where possible.

## Example Use Cases

- Physics simulations (classical mechanics, relativity, quantum systems)
- Mathematical modeling (differential equations, optimization problems)
- Data transformations (tensor operations, signal processing)
- Economic simulations (agent-based models, game theory)
- Any domain that can be expressed mathematically