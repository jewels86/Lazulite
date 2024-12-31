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
			string content = args[0];
			StandardTokenizer tokenizer = new StandardTokenizer();

			Ruleset ruleset = new Ruleset(";", "//");
			ruleset.AddTypes(["int", "float", "string", "char", "bool", "void"]);
			ruleset.AddTypeLiteral("int", TokenizationFunctions.StandardIntegerLiteralRegex);
			ruleset.AddTypeLiteral("float", TokenizationFunctions.StandardFloatLiteralRegex);
			ruleset.AddTypeLiteral("string", TokenizationFunctions.StandardStringLiteralRegex);
			ruleset.AddTypeLiteral("char", TokenizationFunctions.StandardCharLiteralRegex);
			ruleset.AddTypeLiteral("bool", TokenizationFunctions.StandardBoolLiteralRegex);
			ruleset.AddTypeLiteral("void", "void");
			ruleset.AddOperators(TokenizationFunctions.StandardMathOperators);
			ruleset.AddOperators(TokenizationFunctions.StandardComparisonOperators);
			ruleset.AddOperator("=");
			ruleset.SetIdentifier(TokenizationFunctions.StandardIdentifierRegex);
			tokenizer.AddRules(ruleset.Unpack());
			tokenizer.AddRule(TokenizationFunctions.CreateSpaceRule());

			foreach (Token token in tokenizer.Tokenize(content))
			{
				Console.WriteLine($"{token.Value} - {token.Type}");
			}

			var tokens = tokenizer.Tokenize(content);
			Console.WriteLine(string.Join("", tokens.Select(token => $"{token.Value}")));
		}
	}
}