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
	public class ParsingRuleset
	{
		public List<string> LiteralTypes { get; set; } = [];
		public string IdentifierType { get; set; }
		public string BinaryOperatorType { get; set; }
		public string UnaryOperatorType { get; set; }
		public string CommaType { get; set; }
		public string LeftParenthesisType { get; set; }
		public string RightParenthesisType { get; set; }

		public TokenRule IdentifierRule { get; private set; } = null!;
		public ChoiceRule<Token> LiteralRule { get; private set; } = null!;
		public ChoiceRule<Token> FunctionCallRule { get; private set; } = null!;
		public TokenRule OperatorRule { get; private set; } = null!;
		public TokenRule UnaryOperatorRule { get; private set; } = null!;
		public ChoiceRule<Token> FactorRule { get; private set; } = null!;
		public SequenceRule<Token> UnaryExpressionRule { get; private set; } = null!;
		public SequenceRule<Token> BinaryExpressionRule { get; private set; } = null!;
		public ChoiceRule<Token> ExpressionRule { get; private set; } = null!;

		public ParsingRuleset(Dictionary<string, string?> types, List<string>? literalTypes = null) 
		{
			LiteralTypes = literalTypes ?? [];
			IdentifierType = types["identifier"] ?? "identifier";
			BinaryOperatorType = types["operator"] ?? "operator";
			UnaryOperatorType = types["unary-operator"] ?? "unary-operator";
			CommaType = types["comma"] ?? "comma";
			LeftParenthesisType = types["left-parenthesis"] ?? "left-parenthesis";
			RightParenthesisType = types["right-parenthesis"] ?? "right-parenthesis";

			ConstructRules();
		}

		public void ConstructRules()
		{
			IdentifierRule = new TokenRule(IdentifierType, token => new IdentifierAstNode(token.Value));
			LiteralRule = new ChoiceRule<Token>(
				LiteralTypes.Select(type => new TokenRule(type, token => new LiteralAstNode(token.Value, token.Type))).Select(rule => (IGrammarRule<Token>)rule).ToList()
			);
			FunctionCallRule = new ChoiceRule<Token>([
				new SequenceRule<Token>([
					IdentifierRule,
					new TokenRule(LeftParenthesisType, token => null),
					new TokenRule(RightParenthesisType, token => new RepetitionAstNode([]))
				], nodes => new FunctionCallAstNode(nodes[0], ((RepetitionAstNode)nodes[2]).Children)),
				new SequenceRule<Token>([
					IdentifierRule,
					new TokenRule(LeftParenthesisType, token => null),
					null, // to be injected with ExpressionRule
					new TokenRule(RightParenthesisType, token => null)
				], nodes => new FunctionCallAstNode(nodes[0], [nodes[2]])),
				new SequenceRule<Token>([
					IdentifierRule,
					new TokenRule(LeftParenthesisType, token => null),
					null, // to be injected with ExpressionRule
					new RepetitionRule<Token>(new SequenceRule<Token>([
						new TokenRule(CommaType, token => null),
						null // to be injected with ExpressionRule
					], nodes => nodes[1])),
					new TokenRule(RightParenthesisType, token => null)
				], nodes => new FunctionCallAstNode(nodes[0], ((RepetitionAstNode)nodes[3]).Children.Append(nodes[2]).ToList()))
			]);
			OperatorRule = new TokenRule(BinaryOperatorType, token => new OperatorAstNode(token.Value));
			UnaryOperatorRule = new TokenRule(UnaryOperatorType, token => new OperatorAstNode(token.Value));
			FactorRule = new ChoiceRule<Token>([
				FunctionCallRule,
				LiteralRule,
				IdentifierRule,
				new SequenceRule<Token>([
					new TokenRule(LeftParenthesisType, token => null),
					null, // to be injected with ExpressionRule
					new TokenRule(RightParenthesisType, token => null)
				], nodes => nodes[1]),
			]);
			UnaryExpressionRule = new SequenceRule<Token>([
				UnaryOperatorRule,
				FactorRule
			], nodes => new ExpressionAstNode(nodes[1], null, nodes[0]));
			BinaryExpressionRule = new SequenceRule<Token>([
				FactorRule,
				OperatorRule,
				FactorRule
			], nodes => new ExpressionAstNode(nodes[0], nodes[2], nodes[1]));
			ExpressionRule = new ChoiceRule<Token>([
				UnaryExpressionRule,
				BinaryExpressionRule,
				FactorRule
			]);


			var factorSequenceRule = (SequenceRule<Token>)FactorRule.Choices[3];
			factorSequenceRule.Rules[1] = ExpressionRule;

			var functionCallSequenceRule1 = (SequenceRule<Token>)FunctionCallRule.Choices[1];
			functionCallSequenceRule1.Rules[2] = ExpressionRule;
			var functionCallSequenceRule2 = (SequenceRule<Token>)FunctionCallRule.Choices[2];
			functionCallSequenceRule2.Rules[2] = ExpressionRule;
			var functionCallRepetitionRule = (RepetitionRule<Token>)((SequenceRule<Token>)FunctionCallRule.Choices[2]).Rules[3]!;
			var functionCallSequenceRule = (SequenceRule<Token>)functionCallRepetitionRule.Rule;
			functionCallSequenceRule.Rules[1] = ExpressionRule;
		}
	}
}