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
		public static GrammarRuleDelegate CreateParseLiteralRule(string type)
		{
			return (ParserContext ctx, out IAstNode? node) =>
			{
				node = null;

				if (ctx.CurrentToken().Type == type) 
				{
					node = new LiteralAstNode(ctx.CurrentToken().Value, type);
					return true;
				}

				return false;
			};
		}
		
	}
}
