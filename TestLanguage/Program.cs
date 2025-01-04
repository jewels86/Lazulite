using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;
using Lazulite.Parsing;

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

			RecursiveDescentParser<Token> parser = new RecursiveDescentParser<Token>([], null);

			//var expression = 

			var assignment = new GrammarRules.SequenceRule<Token>([
				new GrammarRules.TokenRule("type", t => new AstNodes.TypeAstNode("int")),
				new GrammarRules.TokenRule("identifier", t => new AstNodes.IdentifierAstNode(t.Value)),
				new GrammarRules.TokenRule("assignment-operator", t => null),
				new GrammarRules.TokenRule("int", t => new AstNodes.LiteralAstNode(t.Value, "int")),
			], (nodes) => new AstNodes.StaticAssignmentAstNode(nodes[1], nodes[2], nodes[0]));
			parser.AddRules([assignment]);
			var node = parser.Parse(new ParserContext<Token>(tokens.ToList()));

			node?.Traverse(node => Console.WriteLine(node));
		}
	}
}