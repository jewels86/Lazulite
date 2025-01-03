using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public delegate List<IAstNode> GrammarRuleActionDelgate(IAstNode node);
	public class GrammarRule
	{
		public List<string> Symbols { get; }
		public GrammarRuleActionDelgate? Action { get; }
		
		public GrammarRule(List<string> symbols, GrammarRuleActionDelgate? action = null)
		{
			Symbols = symbols;
			Action = action;
		}
	}
}
