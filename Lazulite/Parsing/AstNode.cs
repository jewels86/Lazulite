using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public class AstNode
	{
		public string Type { get; }
		public string? Value { get; set;}
		public AstNode? Parent { get; set; }
		public List<AstNode> Children { get; } = [];

		public AstNode(string type) 
		{
			Type = type;
		}

		public void AddChild(AstNode child) 
		{
			child.Parent = this;
			Children.Add(child);
		}
		public void RemoveChild(AstNode node)
		{
			if (Children.Contains(node))
			{
				Children.Remove(node);
				node.Parent = null;
			}
		}
		public void ReplaceChild(AstNode old, AstNode replacement)
		{
			if (Children.Contains(old))
			{
				Children[Children.IndexOf(old)] = replacement;
				replacement.Parent = this;
				old.Parent = null;
			}
		}

		public IEnumerable<AstNode> Traverse()
		{
			yield return this;
			foreach (AstNode child in Children) 
			{
				foreach (AstNode descendant in child.Traverse())
				{
					yield return descendant;
				}
			}
		}

		public override string ToString()
		{
			string value = Value == null ? "" : $": {Value}";
			return $"{Type}{value}";
		}
	}
}