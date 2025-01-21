using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
	public class RecursiveDescentParser<T> : IParser<T>, IModularParser<T>
	{
		private readonly List<IGrammarRule<T>> _rules;
		private ParserErrorDelegate<T> _errorHandler;

		public RecursiveDescentParser(List<IGrammarRule<T>> rules, ParserErrorDelegate<T>? errorHandler)
		{
			_rules = rules;
			_errorHandler = errorHandler ?? ((ParserContext<T> ctx, int index) => Console.WriteLine($"Parsing error at index {index}"));
		}

		public void AddRule(IGrammarRule<T> rule)
		{
			_rules.Add(rule);
		}
		public void AddRules(IEnumerable<IGrammarRule<T>> rules)
		{
			_rules.AddRange(rules);
		}

		public void SetErrorHandler(ParserErrorDelegate<T> errorHandler)
		{
			_errorHandler = errorHandler;
		}

		public IAstNode? Parse(ParserContext<T> ctx)
		{
			foreach (var rule in _rules)
			{
				if (rule.Match(ctx, out var node))
				{
					return node;
				}
			}

			_errorHandler(ctx, ctx.Index);
			return null;
		}
	}
}
