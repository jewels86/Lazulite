using Lazulite.Tokenization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public static class ParsingFunctions
	{
		public static GrammarRuleDelegate Assignment()
		{
			return (tokens, index) =>
			{
				ReadOnlySpan<Token> span = new(tokens.ToArray(), index, tokens.Count() - index);

				if (span)
			};
		}
	}
}
