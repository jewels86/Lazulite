using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing.Parsers
{
	public class LR1Parser : IParser
	{
		private readonly Stack<Token> _stack = new();
		private readonly List<GrammarRuleDelegate> _rules = new();

		public IAstNode? Parse(ParserContext context)
		{
			
		}
	}
}
