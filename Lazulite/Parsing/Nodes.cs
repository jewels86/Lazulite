using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public static class Nodes
	{
		public class LiteralAstNode : IAstNode
		{
			public string Value { get; set; }
			public string Type { get; set; }

			public string NodeType => "literal";

			public LiteralAstNode(string value, string type)
			{
				Value = value;
				Type = type;
			}

			public IEnumerable<IAstNode> Traverse()
			{
				yield return this;
			}

			public override string ToString()
			{
				return $"LiteralAstNode(Value: {Value}, Type: {Type})";
			}
		}

		public class BinaryExpressionAstNode : IAstNode
		{
			public string Operator { get; set; }
			public IAstNode Left { get; set; }
			public IAstNode Right { get; set; }

			public string NodeType => "binary-expression";

			public BinaryExpressionAstNode(string op, IAstNode left, IAstNode right)
			{
				Operator = op;
				Left = left;
				Right = right;
			}

			public IEnumerable<IAstNode> Traverse()
			{
				yield return this;
				foreach (var node in Left.Traverse()) yield return node;
				foreach (var node in Right.Traverse()) yield return node;
			}

			public override string ToString()
			{
				return $"BinaryExpressionAstNode(Operator: {Operator}, Left: {Left}, Right: {Right})";
			}
		}

		public class StaticAssignmentAstNode : IAstNode
		{
			public IAstNode Identifier { get; set; }
			public IAstNode Expression { get; set; }
			public IAstNode Type { get; set; }

			public string NodeType => "assignment";

			public StaticAssignmentAstNode(IAstNode type, IAstNode identifier, IAstNode expression)
			{
				Type = type;
				Identifier = identifier;
				Expression = expression;
			}

			public IEnumerable<IAstNode> Traverse()
			{
				yield return this;
				foreach (var node in Identifier.Traverse()) yield return node;
				foreach (var node in Expression.Traverse()) yield return node;
				foreach (var node in Type.Traverse()) yield return node;
			}

			public override string ToString()
			{
				return $"StaticAssignmentAstNode(Type: {Type}, Identifier: {Identifier}, Expression: {Expression})";
			}
		}

		public class FunctionCallAstNode : IAstNode
		{
			public IAstNode FunctionName { get; set; }
			public List<IAstNode> Arguments { get; set; }

			public string NodeType => "function-call";

			public FunctionCallAstNode(IAstNode functionName, List<IAstNode> arguments)
			{
				FunctionName = functionName;
				Arguments = arguments;
			}

			public IEnumerable<IAstNode> Traverse()
			{
				yield return this;
				foreach (var node in FunctionName.Traverse()) yield return node;
				foreach (var node in Arguments.SelectMany(arg => arg.Traverse())) yield return node;
			}

			public override string ToString()
			{
				return $"FunctionCallAstNode(FunctionName: {FunctionName}, Arguments: [{string.Join(", ", Arguments)}])";
			}
		}

		public class IfStatementAstNode : IAstNode
		{
			public IAstNode Condition { get; set; }
			public IAstNode TrueBlock { get; set; }
			public IAstNode? FalseBlock { get; set; }

			public string NodeType => "if-statement";

			public IfStatementAstNode(IAstNode condition, IAstNode trueBlock, IAstNode? falseBlock)
			{
				Condition = condition;
				TrueBlock = trueBlock;
				FalseBlock = falseBlock;
			}

			public IEnumerable<IAstNode> Traverse()
			{
				yield return this;
				foreach (var node in Condition.Traverse()) yield return node;
				foreach (var node in TrueBlock.Traverse()) yield return node;
				if (FalseBlock is not null) foreach (var node in FalseBlock.Traverse()) yield return node;
			}

			public override string ToString()
			{
				return $"IfStatementAstNode(Condition: {Condition}, TrueBlock: {TrueBlock}, FalseBlock: {FalseBlock})";
			}
		}

		public class IdentifierAstNode : IAstNode
		{
			public string Name { get; set; }

			public string NodeType => "identifier";

			public IdentifierAstNode(string name)
			{
				Name = name;
			}

			public IEnumerable<IAstNode> Traverse()
			{
				yield return this;
			}

			public override string ToString()
			{
				return $"IdentifierAstNode(Name: {Name})";
			}
		}

		public class TypeAstNode : IAstNode
		{
			public string Name { get; set; }

			public string NodeType => "type";

			public TypeAstNode(string name)
			{
				Name = name;
			}

			public IEnumerable<IAstNode> Traverse()
			{
				yield return this;
			}

			public override string ToString()
			{
				return $"TypeAstNode(Name: {Name})";
			}
		}
	}
}
