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
			]);
		}
		public static SequenceRule<Token> FunctionCallTokenRule()
		{
			
		}
		public static IGrammarRule<Token> BinaryExpressionTokenRule()
		{
			IGrammarRule<Token> BuildRuleForPrecedence(int precedenceLevel)
			{
				if (precedenceLevel >= PrecedenceLevels.Count)
					return PrimaryExpressionRule();

				var (operators, isRightAssociative) = PrecedenceLevels[precedenceLevel];
				var subExpressionRule = BuildRuleForPrecedence(precedenceLevel + 1);

				return new SequenceRule<Token>(
					new List<IGrammarRule<Token>>
					{
						subExpressionRule,
						new RepetitionRule<Token>(
							new SequenceRule<Token>([
								new TokenRule(OperatorType, t => operators.Contains(t.Value) ? new OperatorAstNode(t.Value) : null), // Operator
								subExpressionRule
							], (nodes) => new ExpressionAstNode(
								null,
								(ExpressionAstNode)nodes[1],
								nodes[0],
								isRightAssociative
							))
						)
					},
					(nodes) =>
					{
						var left = (ExpressionAstNode)nodes[0];
						var repetitions = ((RepetitionAstNode)nodes[1]).Children;

						foreach (var operationNode in repetitions)
						{
							var binaryNode = (ExpressionAstNode)operationNode;
							if (isRightAssociative)
							{
								binaryNode.Left = left;
								left = binaryNode;
							}
							else
							{
								binaryNode.Left = left;
								left = binaryNode;
							}
						}

						return left;
					}
				);
			}

			return BuildRuleForPrecedence(0);
		}
		public static ChoiceRule<Token> ExpressionTokenRule()
		{
			return new ChoiceRule<Token>([
				PrimaryExpressionRule(),
				BinaryExpressionTokenRule(),
				//FunctionCallTokenRule(),
				/*new SequenceRule<Token>([
					new TokenRule("left-parenthesis", t => null),
					ExpressionTokenRule(),
					new TokenRule("right-parenthesis", t => null)
				], (nodes) => nodes[1])*/
			]);
		}
	}
}
