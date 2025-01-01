using Lazulite.Tokenization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Lazulite.Parsing.Nodes;

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
					ctx.Consume();
					return true;
				}

				return false;
			};
		}
		public static GrammarRuleDelegate CreateParseTypeRule(string type)
		{
			return (ParserContext ctx, out IAstNode? node) =>
			{
				node = null;

				if (ctx.CurrentToken().Type == type)
				{
					node = new TypeAstNode(ctx.CurrentToken().Value);
					ctx.Consume();
					return true;
				}

				return false;
			};
		}
		public static GrammarRuleDelegate CreateParseIdentifierRule(string type)
		{
			return (ParserContext ctx, out IAstNode? node) =>
			{
				node = null;

				if (ctx.CurrentToken().Type == type)
				{
					node = new IdentifierAstNode(ctx.CurrentToken().Value);
					ctx.Consume();
					return true;
				}

				return false;
			};
		}
		public static GrammarRuleDelegate CreateParseBinaryExpressionRule(string opType)
		{
			return (ParserContext ctx, out IAstNode? node) =>
			{
				node = null;



				return false;
			};
		} // !!
		public static GrammarRuleDelegate CreateParseStaticAssignmentRule(string opType, GrammarRuleDelegate parseType, GrammarRuleDelegate parseIdentifier, GrammarRuleDelegate parseExpression)
		{
			return (ParserContext ctx, out IAstNode? node) =>
			{
				node = null;

				if (!parseType(ctx, out var typeNode))
					return false;

				if (!parseIdentifier(ctx, out var identifierNode))
					return false;

				var token = ctx.CurrentToken();
				if (token.Type != opType)
					return false;

				if (!parseExpression(ctx, out var expressionNode))
					return false;

				node = new StaticAssignmentAstNode(typeNode!, identifierNode!, expressionNode!);
				return true;

			};
		}
		public static GrammarRuleDelegate CreateParseExpressionRule(List<GrammarRuleDelegate> subRules)
		{
			return (ParserContext ctx, out IAstNode? node) =>
			{
				node = null;

				foreach (var rule in subRules)
				{
					if (rule(ctx, out node))
						return true;
				}

				return false;
			};
		}
	}
}
