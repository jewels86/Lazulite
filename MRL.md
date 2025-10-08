# MRL Language Specification

MRL (Mathematical Representation Language) is a high-level language for describing mathematical systems. It compiles to WASM modules.

## Core Concepts

### Interface Types (`itype`)

Interface types define contracts that implementations must satisfy:

```
vector_t: itype tensor_t = {
    ndims: iconstant int = 1,
    length: int,
    magnitude: scalar_t,
    normalized: vector_t,
    dot(other: same vector_t) -> scalar_t,
}
```

### Complete Types

The `complete` keyword marks types that can be instantiated:

```
vector3: complete itype vector_t = {
    # Full implementation
}
```

---

## Type Modifiers

| Modifier | Scope | Mutability | Description |
|----------|-------|------------|-------------|
| `static` | Type | - | Belongs to the type itself |
| `istatic` | Implementation | - | Belongs to the implementation |
| `constant` | Type | Immutable | `static readonly` |
| `iconstant` | Implementation | Immutable | `istatic readonly` |
| `readonly` | Instance | Immutable | Cannot be modified after initialization |
| `dynamic` | Instance | Computed | Value determined at runtime/access |
| `required` | Instance | - | Must be provided at construction |
| `nullable` | Instance | - | Can be null |
| `specific` | Implementation | - | Specific to this implementation, not inherited |
| `inplace` | Method/Operator | - | Modifies receiver instead of creating new instance |

---

## Type Relationships

### `alike`

Indicates a type can substitute for another without strict inheritance:

```
vector4: complete itype vector_t alike vector3 {
    # Can be used anywhere vector3 is expected
    # Inherits vector3's methods by default
    # May have additional members (like w component)
}
```

**Important**: Inherited operators preserve hidden fields from left operand only. Provide explicit overloads to preserve both:

```
# Inherited - preserves this.w, not other.w
operator this +(other: vector3) -> vector3

# Explicit overload - preserves both
operator this +(other: vector4) -> vector4 = {
    return this with {
        x = this.x + other.x,
        y = this.y + other.y,
        z = this.z + other.z,
        w = this.w + other.w,
    };
}
```

### `same`

In method signatures, means "same type as implementation":

```
dot(other: same vector_t) -> scalar_t
# In vector3: dot(other: vector3) -> scalar_t
# In vector4: dot(other: vector4) -> scalar_t
```

---

## Operators

### Definition

Use `operator` keyword with `this` to indicate receiver:

```
operator this +(other: vector3) -> vector3 = {
    return this with {
        x = this.x + other.x,
        y = this.y + other.y,
        z = this.z + other.z,
    };
}
```

### In-place Operations

Use `inplace` keyword:

```
operator this +=(other: vector3) inplace = {
    this.x := this.x + other.x;
    this.y := this.y + other.y;
    this.z := this.z + other.z;
}
```

### Constructors

Combine `new` with `operator`:

```
operator new this(x: scalar_t, y: scalar_t, z: scalar_t) -> vector3 = {
    this.x = x;
    this.y = y;
    this.z = z;
}
```

---

## Import System

### Basic Import

Imports bring an entire MRL file into the current workspace:

```
import "vectors.mrl";
import "physics.mrl";
```

All types, functions, and declarations from the imported file become available in the current file.

**Note**: Namespacing and selective imports may be added in future versions.

---

## Error Handling

### Assert

Runtime assertion that halts execution if condition is false:

```
assert condition, "error message";
```

**Behavior:**
- Evaluates condition at runtime
- If false, throws runtime error with provided message
- If true, execution continues normally

**Example:**
```
acceleration(b: body_t, s: state) -> vector3 = {
    assert b.mass > 0, "Body mass must be positive";
    assert s.bodies.length > 0, "State must contain at least one body";
    
    let acc = new vector3(0.0, 0.0, 0.0);
    for other in s.bodies.where(o => o != b) {
        acc += gravity(b, other);
    }
    return acc;
}
```

**Future consideration**: Try-catch blocks may be added for more sophisticated error handling.

---

## Control Flow

### While Loops

Standard while loop with condition:

```
while condition {
    # statements
}
```

**Example:**
```
find_equilibrium(system: state, tolerance: scalar_t) -> state = {
    let current = system;
    let delta = tolerance + 1.0;
    
    while delta > tolerance {
        let next = simulate_step(current, 0.01);
        delta = (next.kinetic_energy() - current.kinetic_energy()).abs();
        current := next;
    }
    
    return current;
}
```

### For Loops

For loops are syntactic sugar for `.foreach()`:

```
for item in collection {
    # statements
}
```

**Desugars to:**
```
collection.foreach(item => {
    # statements
});
```

**Example:**
```
total_momentum(bodies: body_t[]) -> vector3 = {
    let total = new vector3(0.0, 0.0, 0.0);
    for body in bodies {
        total += body.momentum();
    }
    return total;
}
```

**Note**: The `in` keyword is specifically for for-loops and is not a general operator.

---

## Control Flow Keywords

| Keyword | Purpose |
|---------|---------|
| `break` | Exit current loop |
| `continue` | Skip to next loop iteration |
| `return` | Return value from function |

**Example with break/continue:**
```
find_first_heavy_body(bodies: body_t[], threshold: scalar_t) -> nullable body_t = {
    for body in bodies {
        if body.mass < 0 {
            continue;  # Skip invalid bodies
        }
        if body.mass > threshold {
            return body;  # Found it
        }
    }
    return null;  # Not found
}
```

## Preservation and Modification

### Assignment vs Modification

**Assignment (`=`)**: Replaces reference with new object
```
x = new vector3(1, 2, 3);  # x points to new vector3
```

**Modification (`:=`)**: Updates existing object in place
```
x := new vector3(1, 2, 3);  # Modifies x's fields
```

**Rule**: Use `:=` by default. It preserves hidden fields and is more efficient.

### `with` Keyword

Creates new instance with modified fields. **Preserves all hidden fields including from subtypes**:

```
new_vector = existing_vector with { x = 10 }
# If existing_vector is vector4 used as vector3, w is preserved
```

### `out` Keyword

Indicates parameter should be modified in place. Preserves hidden members:

```
transform(out position: vector3, out velocity: vector3, time: scalar_t) = {
    position := position + (velocity * time);  # OK
    position = new vector3(...);  # COMPILER ERROR
}
```

**Compiler enforcement**: Cannot reassign `out` parameters with `=`.

### `preserves` Keyword

Indicates function returns the same instance passed in:

```
normalize(vec: vector3) -> preserves vec = {
    let magnitude = vec.magnitude;
    vec := vec / magnitude;
    return vec;
}

# Allows chaining:
result = normalize(my_vector).scale(2.0);
```

---

## Type Casting

### Cast from Other Types

```
operator from int(value: int) -> scientific_number = {
    return new scientific_number(value as float, 0);
}
```

### Cast to Other Types

```
operator to int() -> int = {
    return (this.mantissa * pow(10, this.exponent)) as int;
}
```

Casting is automatically invoked during type conversions.

### Custom Modification Operators

Define how types handle `:=`:

```
operator this :=(value: int) inplace = {
    this.mantissa := value as float;
    this.exponent := 0;
}
```

---

## Type Reflection

Runtime type information via `typeof`:

```
typeof(a).new(...)   # Create instance of same type as 'a'
typeof(a).clone()    # Clone instance maintaining type
```

**Best practice**: Use `typeof` in generic code to avoid hardcoding concrete types.

---

## Compiler Guarantees

The compiler verifies:
- `out` parameters are not reassigned
- `preserves` functions return correct instance
- Type conversions are valid
- Hidden field preservation in `with` and `:=`

---

## Compiler Inference

The compiler can automatically infer certain keywords and generate boilerplate to reduce manual annotation.

### `preserves` Inference

The compiler infers `preserves` for a return value when:
1. The return expression traces back to a parameter
2. That parameter is only modified via `with` or `:=` (never reassigned with `=`)
3. The return value is that parameter or derived from it through preservation-safe operations

```
operator this +(other: vector3) -> vector3 = {
    return this with { x = this.x + other.x, ... };
} # Compiler infers: -> preserves this
```

### `out` Inference

The compiler infers `out` for parameters when:
1. The parameter is modified with `:=`
2. The parameter is never reassigned with `=`
3. The modifications are visible to the caller (not shadowed)

```
transform(position: vector3, velocity: vector3, time: scalar_t) = {
    position := position + (velocity * time);
} # Compiler infers: out position
```

### Auto-generated Modification Operators

For interface types without explicit `:=` operators, the compiler generates a default implementation:

**Generated behavior:**
```
operator this :=(other: same_interface_t) inplace = {
    # For each field in the interface:
    this.field := other.field;
    # Hidden fields (from subtypes) are preserved
}
```

This allows modification between any types implementing the same interface:
```
body: body_t = create_body_with_metadata(...);
body := new normal_body(mass, position, velocity);
# Sets mass, position, velocity
# Preserves metadata fields if body has them
```

**When to write custom `:=` operators:**
- Type requires special invariant maintenance (like `scientific_number` normalization)
- Conversion logic is non-trivial
- Performance-critical paths need optimization
- Default field-by-field copying is incorrect for the type's semantics

## Design Philosophy

1. **Readability**: Natural language and mathematical notation
2. **Flexibility**: Interface-based design for multiple implementations
3. **Type Safety**: Strong typing with runtime information when needed
4. **Data Integrity**: Preservation mechanisms prevent hidden field loss
5. **Modularity**: Separation of concerns
6. **Reusability**: Interface-based code works with any valid implementation

---

## Compilation Target

Compiles to WASM modules for portability. Compiler resolves interface types, generates efficient code, and handles dynamic property computation at compile time where possible.