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
		public static string BinaryOperatorType { get; set; } = "operator";
		public static string UnaryOperatorType { get; set; } = "unary-operator";
		public static string CommaType { get; set; } = "comma";
		public static string LeftParenthesisType { get; set; } = "left-parenthesis";
		public static string RightParenthesisType { get; set; } = "right-parenthesis";

		public static IGrammarRule<Token> IdentifierRule = new TokenRule(IdentifierType, token => new IdentifierAstNode(token.Value));
		public static IGrammarRule<Token> LiteralRule = new ChoiceRule<Token>(
			LiteralTypes.Select(type => new TokenRule(type, token => new LiteralAstNode(token.Value, token.Type))).Select(rule => (IGrammarRule<Token>)rule).ToList()
		);
		public static SequenceRule<Token> FunctionCallRule = new SequenceRule<Token>([
			IdentifierRule, //0 
			new TokenRule(LeftParenthesisType, token => null), //1
			new OptionalRule<Token>(new SequenceRule<Token>([ //2
				null,
			], nodes => nodes[0])),
			new RepetitionRule<Token>(new SequenceRule<Token>([ //3
				new TokenRule(CommaType, token => null),
				null
			], nodes => nodes[1])),
			new TokenRule(RightParenthesisType, token => null)
		], nodes =>
		{
			Console.WriteLine($"Function {nodes[0]}");

			List<IAstNode> args = [];
			if (nodes[2] is not null)
			{
				args.Add(nodes[2]);
				args.AddRange(((RepetitionAstNode)nodes[3]).Children);
			}
			return new FunctionCallAstNode(nodes[0], args);
		});
		public static IGrammarRule<Token> OperatorRule = new TokenRule(BinaryOperatorType, token => new OperatorAstNode(token.Value));
		public static IGrammarRule<Token> UnaryOperatorRule = new TokenRule(UnaryOperatorType, token => new OperatorAstNode(token.Value));
		public static ChoiceRule<Token> FactorRule = new ChoiceRule<Token>([
			LiteralRule,
			FunctionCallRule,
			new SequenceRule<Token>([
				new TokenRule(LeftParenthesisType, token => null),
				null,
				new TokenRule(RightParenthesisType, token => null)
			], nodes => nodes[1]),
			IdentifierRule
		]); 
		public static IGrammarRule<Token> UnaryExpressionRule = new SequenceRule<Token>([
			UnaryOperatorRule,
			FactorRule
		], nodes => new ExpressionAstNode(nodes[1], null, nodes[0]));
		public static IGrammarRule<Token> BinaryExpressionRule = new SequenceRule<Token>([
			FactorRule,
			OperatorRule,
			FactorRule
		], nodes => new ExpressionAstNode(nodes[0], nodes[2], nodes[1]));

		public static IGrammarRule<Token> ExpressionRule = null!;

		public static void InitializeRules()
		{
			ExpressionRule = new ChoiceRule<Token>([
				UnaryExpressionRule,
				BinaryExpressionRule,
				FactorRule
			]);

			((SequenceRule<Token>)FactorRule.Choices[2]).Rules[1] = ExpressionRule;
			((SequenceRule<Token>)((OptionalRule<Token>)FunctionCallRule.Rules[2]!).Rule).Rules[0] = ExpressionRule;
			((SequenceRule<Token>)((RepetitionRule<Token>)FunctionCallRule.Rules[3]!).Rule).Rules[1] = ExpressionRule;

		}
		public static IGrammarRule<Token> ExpressionSequenceRule = new RepetitionRule<Token>(ExpressionRule);
		public static IGrammarRule<Token> CreateExpressionRule()
		{
			return new SequenceRule<Token>([ExpressionRule], nodes =>
			{
				return nodes[0];
			});
		}
	}
}