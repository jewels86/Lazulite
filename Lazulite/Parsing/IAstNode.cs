using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public interface IAstNode
	{
		public string NodeType { get; }

		public void Traverse(Action<IAstNode> action);

		public IAstNode this[int index] { get; set; }
	}
}
