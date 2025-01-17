using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace Lazulite.Parsing.Lpn
{
	public static class Nodes
	{
		public class MetadataAstNode : IAstNode
		{
			public string Key { get; }
			public List<IAstNode> Operands { get; }

			public string NodeType => "metadata";

			public MetadataAstNode(string name, List<IAstNode> operands)
			{
				Key = name;
				Operands = operands;
			}

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				foreach (var operand in Operands)
				{
					operand.Traverse(action);
				}
			}
		}
		public class DeclarationAstNode : IAstNode
		{
			public IAstNode Identifier { get; set; }
			public IAstNode Rule { get; set; }
			public IAstNode? Transformer { get; set; }

			public DeclarationAstNode(IAstNode identifier, IAstNode rule, IAstNode? transformer)
			{
				Identifier = identifier;
				Rule = rule;
				Transformer = transformer;
			}

			public string NodeType => "declaration";

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
			}

			public override string ToString()
			{
				return $"DeclarationAstNode(Identifier: {Identifier}, Rule: {Rule}, Transformer: {Transformer})";
			}

			public IAstNode this[int index]
			{
				get
				{
					if (index == 0) return Identifier;
					else if (index == 1) return Rule;
					else if (index == 2 && Transformer is not null) return Transformer;
					else throw new IndexOutOfRangeException();
				}
				set
				{
					if (index == 0) Identifier = value;
					else if (index == 1) Rule = value;
					else if (index == 2) Transformer = value;
					else throw new IndexOutOfRangeException();
				}
			}
		}
		public class ScriptDefinitionAstNode : IAstNode
		{
			public IAstNode Type { get; set; }
			public IAstNode Identifier { get; set; }
			public IAstNode Expression { get; set; }

			public ScriptDefinitionAstNode(IAstNode type, IAstNode identifier, IAstNode expression)
			{
				Type = type;
				Identifier = identifier;
				Expression = expression;
			}

			public string NodeType => "script-declaration";

			public void Traverse(Action<IAstNode> action)
			{
				action(this);
				Type.Traverse(action);
				Identifier.Traverse(action);
				Expression.Traverse(action);
			}

			public override string ToString()
			{
				return $"ScriptDefinitionAstNode(Type: {Type}, Identifier: {Identifier}, Expression: {Expression})";
			}

			public IAstNode this[int index]
			{
				get
				{
					if (index == 0) return Type;
					else if (index == 1) return Identifier;
					else if (index == 2) return Expression;
					else throw new IndexOutOfRangeException();
				}
				set
				{
					if (index == 0) Type = value;
					else if (index == 1) Identifier = value;
					else if (index == 2) Expression = value;
					else throw new IndexOutOfRangeException();
				}
			}
		}
	}
}
