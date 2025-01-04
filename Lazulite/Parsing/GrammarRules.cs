using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing
{
	public static class GrammarRules
	{
		public class SequenceRule<T> : IGrammarRule<T>
		{
			private readonly List<IGrammarRule<T>> _rules;
			private readonly Func<List<IAstNode>, IAstNode> _transform;

			public SequenceRule(List<IGrammarRule<T>> rules, Func<List<IAstNode>, IAstNode> transform)
			{
				_rules = rules;
				_transform = transform;
			}

			public bool Match(ParserContext<T> ctx, out IAstNode? node)
			{
				var startIndex = ctx.Index;
				var matchedNodes = new List<IAstNode>();

				foreach (var rule in _rules)
				{
					if (!rule.Match(ctx, out var subNode))
					{
						ctx.Restore(startIndex);
						node = null;
						return false;
					}
					matchedNodes.Add(subNode!);
				}

				node = _transform(matchedNodes);
				return true;
			}
		}

		public class TokenRule : IGrammarRule<Token>
		{
			private readonly string _tokenType;
			private readonly Func<Token, IAstNode?> _action;

			public TokenRule(string tokenType, Func<Token, IAstNode?> action)
			{
				_tokenType = tokenType;
				_action = action;
			}

			public bool Match(ParserContext<Token> ctx, out IAstNode? node)
			{
				var token = ctx.Current();
				if (token.Type == _tokenType)
				{
					node = _action(token);
					ctx.Consume();
					return true;
				}
				node = null;
				return false;
			}
		}

		public class OptionalRule<T> : IGrammarRule<T>
		{
			private readonly IGrammarRule<T> _rule;

			public OptionalRule(IGrammarRule<T> rule)
			{
				_rule = rule;
			}

			public bool Match(ParserContext<T> ctx, out IAstNode? node)
			{
				if (_rule.Match(ctx, out var subNode))
				{
					node = subNode;
					return true;
				}
				node = null;
				return true;
			}
		}

		public class RepetitionRule<T> : IGrammarRule<T>
		{
			private readonly IGrammarRule<T> _rule;

			public RepetitionRule(IGrammarRule<T> rule)
			{
				_rule = rule;
			}

			public bool Match(ParserContext<T> ctx, out IAstNode? node)
			{
				var matchedNodes = new List<IAstNode>();
				while (_rule.Match(ctx, out var subNode))
				{
					matchedNodes.Add(subNode!);
				}
				node = new AstNodes.RepetitionAstNode(matchedNodes);
				return matchedNodes.Count > 0;
			}
		}

		public class ChoiceRule<T> : IGrammarRule<T>
		{
			private readonly List<IGrammarRule<T>> _choices;

			public ChoiceRule(List<IGrammarRule<T>> choices)
			{
				_choices = choices;
			}

			public bool Match(ParserContext<T> ctx, out IAstNode? node)
			{
				foreach (var choice in _choices)
				{
					if (choice.Match(ctx, out node))
					{
						return true;
					}
				}
				node = null;
				return false;
			}
		}
	}
}
