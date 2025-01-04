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
		}

		public class StaticAssignmentAstNode : IAstNode
		{
			public IdentifierAstNode Identifier { get; }
			public ExpressionAstNode Value { get; }
			public TypeAstNode Type { get; }

			public string NodeType => "Static Assignment";

			public StaticAssignmentAstNode(IAstNode identifier, IAstNode value, IAstNode type)
			{
				if (identifier is not IdentifierAstNode && value is not ExpressionAstNode && type is not TypeAstNode)
				{
					throw new ArgumentException($"Value must be an expression and type must be a type, got {value} and {type}");
				}
				Identifier = (IdentifierAstNode)identifier;
				Value = (ExpressionAstNode)value;
				Type = (TypeAstNode)type;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				Value.Traverse(action);
				Type.Traverse(action);
			}

			public override string ToString()
			{
				return $"StaticAssignmentAstNode(Value: {Value}, Type: {Type})";
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
		}

		public class ExpressionAstNode : IAstNode
		{
			public IAstNode? Left { get; }
			public IAstNode? Right { get; }
			public IAstNode? Operator { get; }
			public bool Binary => Left != null && Right != null;
			public bool RightAssociative { get; }

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
		}
	}
}
