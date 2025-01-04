using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;
using static Lazulite.Parsing.GrammarRules;
using static Lazulite.Parsing.AstNodes;
using Lazulite.Tokenization;

namespace Lazulite.Parsing
{
	public static class ParsingFunctions
	{
		public static List<string> LiteralTypes { get; set; } = [];
		public static string IdentifierType { get; set; } = "identifier";
		public static string OperatorType { get; set; } = "operator";
		public static List<(string[], bool)> PrecedenceLevels { get; set; } = [];

		public static ChoiceRule<Token> PrimaryExpressionRule()
		{
			var literalRules = LiteralTypes
			.Select(type => new TokenRule(type, t => new LiteralAstNode(t.Value, type)))
			.ToList<IGrammarRule<Token>>();

			return new ChoiceRule<Token>([
				new TokenRule(IdentifierType, t => new IdentifierAstNode(t.Value)),
				new ChoiceRule<Token>(literalRules),
				new SequenceRule<Token>([
					new TokenRule("left-parenthesis", t => null),
					ExpressionTokenRule(),
					new TokenRule("right-parenthesis", t => null)
				], (nodes) => nodes[1])
			]);
		}
		public static SequenceRule<Token> FunctionCallTokenRule()
		{
			return new SequenceRule<Token>([
				new TokenRule("identifier", t => new IdentifierAstNode(t.Value)),
				new TokenRule("left-parenthesis", t => null),
				new OptionalRule<Token>(ExpressionTokenRule()),
				new RepetitionRule<Token>(new SequenceRule<Token>([
					new TokenRule("comma", t => null),
					ExpressionTokenRule()
				], (nodes) => nodes[1])),
				new TokenRule("right-parenthesis", t => null)
			], (nodes) => nodes[1]);
		}
		public static IGrammarRule<Token> BinaryExpressionTokenRule()
		{
			
		}
		public static ChoiceRule<Token> ExpressionTokenRule()
		{
			List<IGrammarRule<Token>> literalRules = [];
			foreach (var type in LiteralTypes)
			{
				literalRules.Add(new TokenRule(type, t => new LiteralAstNode(t.Value, type)));
			}
			return new ChoiceRule<Token>([
				PrimaryExpressionRule(),
				BinaryExpressionTokenRule(),
				FunctionCallTokenRule(),
				new SequenceRule<Token>([
					new TokenRule("left-parenthesis", t => null),
					ExpressionTokenRule(),
					new TokenRule("right-parenthesis", t => null)
				], (nodes) => nodes[1])
			]);
		}
	}
}
