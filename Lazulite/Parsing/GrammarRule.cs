using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public delegate List<IAstNode> GrammarRuleActionDelegate(IAstNode node);

	public interface IGrammarRule<T>
	{
		public bool Match(ParserContext<T> ctx, out IAstNode? node);
	}
}