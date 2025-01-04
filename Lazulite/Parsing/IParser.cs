using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public delegate void ParserErrorDelegate<T>(ParserContext<T> ctx, int index);
	public interface IParser<T>
	{
		public void AddRule(IGrammarRule<T> rule);
		public void AddRules(IEnumerable<IGrammarRule<T>> rules);

		public IAstNode? Parse(ParserContext<T> ctx);
	}
}