using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing.Lpn
{
	public class LpnFile(string name, List<Token> tokens, IAstNode ast)
	{
		public string Name { get; } = name;
		public List<Token> Tokens { get; } = tokens;
		public IAstNode Ast { get; } = ast;
	}
}
