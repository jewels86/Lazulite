using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;
using Lazulite.Tokenization.Tokenizers;

namespace TestLanguage
{
	public static class Program
	{
		public static void Main(string[] args)
		{
			string content = File.ReadAllText(args[0]);
			StandardTokenizer tokenizer = new StandardTokenizer();
			TokenizationRuleset ruleset = new Ruleset(";", "//");

			ruleset.AddKeywords(TokenizationFunctions.StandardKeywords);
			ruleset.AddOperators(TokenizationFunctions.StandardMathOperators);
			ruleset.AddOperators(TokenizationFunctions.StandardComparisonOperators);
			ruleset.AddOperators(TokenizationFunctions.StandardLogicalOperators);
			ruleset.AddOperators(TokenizationFunctions.StandardAssignmentOperators);
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

			foreach (Token token in tokenizer.Tokenize(content))
			{
				Console.WriteLine($"{token.Value} - {token.Type}");
			}

			var tokens = tokenizer.Tokenize(content);
			Console.WriteLine(string.Join("", tokens.Select(token => $"{token.Value}")));
		}
	}
}