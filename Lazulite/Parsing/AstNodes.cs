using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public static class AstNodes
	{
		public class LiteralAstNode : IAstNode
		{
			public string Value { get; }
			public string Type { get; }

			public string NodeType => "Literal";

			public LiteralAstNode(string value, string type)
			{
				Value = value;
				Type = type;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
			}

			public override string ToString()
			{
				return $"LiteralAstNode(Value: {Value}, Type: {Type})";
			}

			public IAstNode this[int index] 
			{ 
				get => throw new Exception("Cannot get value of literal node");
				set => throw new Exception("Cannot set value of literal node");
			}
		}

		public class IdentifierAstNode : IAstNode
		{
			public string Name { get; }

			public string NodeType => "Identifier";

			public IdentifierAstNode(string name)
			{
				Name = name;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
			}

			public override string ToString()
			{
				return $"IdentifierAstNode(Name: {Name})";
			}

			public IAstNode this[int index]
			{
				get => throw new Exception("Cannot get value of identifier node");
				set => throw new Exception("Cannot set value of identifier node");
			}
		}

		public class RepetitionAstNode : IAstNode
		{
			public List<IAstNode> Children { get; }

			public string NodeType => "Repetition";

			public RepetitionAstNode(List<IAstNode> children)
			{
				Children = children;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				foreach (var child in Children)
				{
					child.Traverse(action);
				}
			}

			public override string ToString()
			{
				return $"RepetitionAstNode(Children: [{string.Join(", ", Children)}])";
			}

			public IAstNode this[int index]
			{
				get => Children[index];
				set => Children[index] = value;
			}
		}

		public class StaticAssignmentAstNode : IAstNode
		{
			public IdentifierAstNode Identifier { get; set; }
			public IAstNode Value { get; set; }

			public string NodeType => "Static Assignment";

			public StaticAssignmentAstNode(IAstNode identifier, IAstNode value)
			{
				if (identifier is not IdentifierAstNode && value is not ExpressionAstNode)
				{
					throw new ArgumentException($"Value must be an expression, got {value}");
				}
				Identifier = (IdentifierAstNode)identifier;
				Value = value;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				Value.Traverse(action);
				Identifier.Traverse(action);
			}

			public override string ToString()
			{
				return $"StaticAssignmentAstNode(Identifier: {Identifier}, Value: {Value})";
			}

			public IAstNode this[int index]
			{
				get
				{
					if (index == 0) return Identifier;
					else if (index == 1) return Value;
					else throw new IndexOutOfRangeException();
				}
				set
				{
					if (index == 0) Identifier = (IdentifierAstNode)value;
					else if (index == 1) Value = value;
					else throw new IndexOutOfRangeException();
				}
			}
		}

		public class StaticDeclarationAstNode : IAstNode
		{

			public IAstNode Type { get; set; }
			public IAstNode Assignment { get; set; }

			public string NodeType => "Static Declaration";

			public StaticDeclarationAstNode(IAstNode type, IAstNode assignment)
			{
				if (type is not TypeAstNode && assignment is not StaticAssignmentAstNode)
				{
					throw new ArgumentException($"Type must be a type, got {type}");
				}
				Type = type;
				Assignment = assignment;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				Type.Traverse(action);
				Assignment.Traverse(action);
			}

			public override string ToString()
			{
				return $"StaticDeclarationAstNode(Type: {Type}, Assignment: {Assignment})";
			}

			public IAstNode this[int index]
			{
				get
				{
					if (index == 0) return Type;
					else if (index == 1) return Assignment;
					else throw new IndexOutOfRangeException();
				}
				set
				{
					if (index == 0) Type = value;
					else if (index == 1) Assignment = value;
					else throw new IndexOutOfRangeException();
				}
			}
		}

		public class TypeAstNode : IAstNode
		{
			public string Type { get; }

			public string NodeType => "Type";

			public TypeAstNode(string type)
			{
				Type = type;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
			}

			public override string ToString()
			{
				return $"TypeAstNode(Type: {Type})";
			}

			public IAstNode this[int index]
			{
				get => throw new Exception("Cannot get value of type node");
				set => throw new Exception("Cannot set value of type node");
			}
		}

		public class ExpressionAstNode : IAstNode
		{
			public IAstNode? Left { get; set; }
			public IAstNode? Right { get; set; }
			public IAstNode? Operator { get; set; }
			public bool Binary => Left != null && Right != null;
			public bool RightAssociative { get; set; }

			public ExpressionAstNode(IAstNode? left, IAstNode? right, IAstNode? op, bool rightAssociative = false)
			{
				Left = left;
				Right = right;
				Operator = op;
				RightAssociative = rightAssociative;
			}

			public string NodeType => "Expression";

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				Left?.Traverse(action);
				Right?.Traverse(action);
				Operator?.Traverse(action);
			}

			public override string ToString()
			{
				return $"ExpressionAstNode(Left: {Left}, Right: {Right}, Operator: {Operator})";
			}
			
			public IAstNode this[int index]
			{
				get
				{
					if (index == 0 && Left is not null) return Left;
					else if (index == 1 && Right is not null) return Right;
					else if (index == 2 && Operator is not null) return Operator;
					else throw new IndexOutOfRangeException();
				}
				set
				{
					if (index == 0) Left = value;
					else if (index == 1) Right = value;
					else if (index == 2) Operator = value;
					else throw new IndexOutOfRangeException();
				}
			}
		}

		public class OperatorAstNode : IAstNode
		{
			public string Operator { get; }

			public string NodeType => "Operator";

			public OperatorAstNode(string op)
			{
				Operator = op;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
			}

			public override string ToString()
			{
				return $"OperatorAstNode(Operator: {Operator})";
			}

			public IAstNode this[int index]
			{
				get => throw new Exception("Cannot get value of operator node");
				set => throw new Exception("Cannot set value of operator node");
			}
		}

		public class FunctionCallAstNode : IAstNode
		{
			public IdentifierAstNode Identifier { get; set; }
			public List<IAstNode> Arguments { get; }

			public string NodeType => "Function Call";

			public FunctionCallAstNode(IAstNode identifier, List<IAstNode> arguments)
			{
				if (identifier is not IdentifierAstNode)
				{
					throw new ArgumentException($"Identifier must be an identifier, got {identifier}");
				}
				Identifier = (IdentifierAstNode)identifier;
				Arguments = arguments;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				Identifier.Traverse(action);
				foreach (var argument in Arguments)
				{
					argument.Traverse(action);
				}
			}

			public override string ToString()
			{
				return $"FunctionCallAstNode(Identifier: {Identifier}, Arguments: [{string.Join(", ", Arguments)}])";
			}

			public IAstNode this[int index]
			{
				get
				{
					if (index == 0) return Identifier;
					else if (index > 0 && index <= Arguments.Count) return Arguments[index - 1];
					else throw new IndexOutOfRangeException();
				}
				set
				{
					if (index == 0) Identifier = (IdentifierAstNode)value;
					else if (index > 0 && index <= Arguments.Count) Arguments[index - 1] = value;
					else throw new IndexOutOfRangeException();
				}
			}
		}

		public class ProgramAstNode : IAstNode
		{
			public List<IAstNode> Statements { get; set; }

			public string NodeType => "Program";

			public ProgramAstNode(List<IAstNode> statements)
			{
				Statements = statements;
			}

			public void Traverse(Action<IAstNode> action)
			{
				foreach (var statement in Statements)
				{
					statement.Traverse(action);
				}
			}

			public override string ToString()
			{
				return $"ProgramAstNode(Statements: [{string.Join(", ", Statements)}])";
			}

			public IAstNode this[int index]
			{
				get => Statements[index];
				set => Statements[index] = value;
			}
		}
	}
}
