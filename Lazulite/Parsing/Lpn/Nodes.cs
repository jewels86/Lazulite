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
			public IAstNode Identifier { get; }
			public IAstNode Rule { get; }
			public IAstNode? Transformer { get; }

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
		}

		public class ScriptDefinitionAstNode : IAstNode
		{
			public IAstNode Type { get; }
			public IAstNode Identifier { get; }
			public IAstNode Expression { get; }

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
		}
	}
}
