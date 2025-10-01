# MRL Language Documentation
MRL (Mathematical Representation Language) is a high-level language designed to describe mathematical systems without delving into implementation specifics. 

## Key Features
- **Interface-based Typing**: MRL uses the `itype` keyword to define types, allowing for flexible designs that can be implemented in various ways.
- **Inheritance**: MRL supports inheritance, enabling the creation of complex types based on simpler ones.
- **Natural Language Syntax**: MRL's syntax is designed to be intuitive and easy to read, closely resembling natural language.
  - **`static`**: Used to define values that belong to the type itself rather than implementations or instances of the type.
  - **`istatic`**: Used to define static members that belong to the implementation of the type.
  - **`readonly`**: Indicates that a member cannot be modified after its initial assignment.
  - **`constant`**: Equivalent to `static readonly` or `istatic readonly`, used for defining constants.
  - **`specific`**: Used to define members that are specific to a particular implementation of a type and not it's inheritors.
  - **`operator`**: Used to define custom operators for types, enhancing expressiveness.
  - **`new`**: Used to define constructor operations for types, allowing for easier initialization of instances.
  - **`dynamic`**: Indicates that a member's value will be determined at runtime, allowing for more flexible designs.
  - 