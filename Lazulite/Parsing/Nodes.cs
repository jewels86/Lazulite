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
		}
		public class IfStatementAstNode : IAstNode
		{
			public IAstNode Condition { get; set; }
			public IAstNode TrueBlock { get; set; }
			public IAstNode FalseBlock { get; set; }

			public string NodeType => "if-statement";

			public IfStatementAstNode(IAstNode condition, IAstNode trueBlock, IAstNode falseBlock)
			{
				Condition = condition;
				TrueBlock = trueBlock;
				FalseBlock = falseBlock;
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
		}
	}
}
