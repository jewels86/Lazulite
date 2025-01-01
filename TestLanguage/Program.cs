using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;
using Lazulite.Parsing;
using Lazulite.Parsing.Parsers;
using Lazulite.Tokenization.Tokenizers;

namespace TestLanguage
{
	public static class Program
	{
		public static void Main(string[] args)
		{
			string content = File.ReadAllText(args[0]);
			StandardTokenizer tokenizer = new StandardTokenizer();
			TokenizationRuleset ruleset = new TokenizationRuleset(";", "//");

			ruleset.AddKeywords(TokenizationFunctions.StandardKeywords);
			ruleset.AddOperators(TokenizationFunctions.StandardMathOperators);
			ruleset.AddOperators(TokenizationFunctions.StandardComparisonOperators);
			ruleset.AddOperators(TokenizationFunctions.StandardLogicalOperators);
			ruleset.AddOperators(TokenizationFunctions.StandardAssignmentOperators, "assignment-operator");
			ruleset.AddTypes(["int", "float", "char", "void", "bool", "string"]);
			ruleset.AddTypeLiteral("int", TokenizationFunctions.StandardIntegerLiteralRegex);
			ruleset.AddTypeLiteral("float", TokenizationFunctions.StandardFloatLiteralRegex);
			ruleset.AddTypeLiteral("char", TokenizationFunctions.StandardCharLiteralRegex);
			ruleset.AddTypeLiteral("bool", TokenizationFunctions.StandardLowercaseBooleanLiteralRegex);
			ruleset.AddTypeLiteral("string", TokenizationFunctions.StandardStringLiteralRegex);
			ruleset.SetIdentifier(TokenizationFunctions.StandardIdentifierRegex);

			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"#include\s*<.*?>", "preprocessor-directive"));
			tokenizer.AddRules(ruleset.Unpack());
			tokenizer.AddRules(TokenizationFunctions.CreateParenthesisRules());
			tokenizer.AddRules(TokenizationFunctions.CreateBraceRules());
			tokenizer.AddRules(TokenizationFunctions.CreateBracketRules());
			tokenizer.AddRules(TokenizationFunctions.CreateWhitespaceRules());
			tokenizer.AddRule(TokenizationFunctions.CreateCommaRule());

			tokenizer.AddPostProcessor(TokenizationFunctions.CreatePostProcessorFilterFromType("space", false));

			var tokens = tokenizer.Tokenize(content);

			foreach (var token in tokens)
			{
				Console.WriteLine(token);
			}

			ParserContext context = new ParserContext(tokens);
			RecursiveDescentParser parser = new();

			var parseIntLiteral = ParsingFunctions.CreateParseLiteralRule("int");
			var parseType = ParsingFunctions.CreateParseTypeRule("type");
			var parseIdentifier = ParsingFunctions.CreateParseIdentifierRule("identifier");
			var parseExpression = ParsingFunctions.CreateParseExpressionRule([parseIntLiteral, parseIdentifier]);
			var parseAssignment = ParsingFunctions.CreateParseStaticAssignmentRule("assignment-operator", parseType, parseIdentifier, parseExpression);

			parser.AddRules([parseAssignment, parseExpression, parseIntLiteral, parseIdentifier, parseType]);

			IAstNode? tree = parser.Parse(context);
			if (tree is not null) foreach (var node in tree.Traverse())
			{
				Console.WriteLine(node);
			}
		}
	}
}