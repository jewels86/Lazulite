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
	}
}
