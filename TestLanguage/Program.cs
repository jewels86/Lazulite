using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace TestLanguage
{
	public static class Program
	{
		public static void Main(string[] args)
		{
			string content = args[0];
			InputSplittingTokenizer tokenizer = new();

			Ruleset ruleset = new(";", "//");
			ruleset.AddTypes(["int", "string", "char", "float"]);
			ruleset.AddTypeLiteral("int-literal", @"\d+");
			ruleset.AddTypeLiteral("string-literal", TokenizationFunctions.StandardStringLiteralRegex);
			ruleset.AddTypeLiteral("char-literal", TokenizationFunctions.StandardCharLiteralRegex);
			ruleset.AddOperators(["+", "-", "*", "/", "="]);
			ruleset.SetIdentifier(TokenizationFunctions.StandardIdentifierRegex);
			tokenizer.AddRules(ruleset.UnpackForSplitInput().ToArray());
			tokenizer.SetPostInputSplitter(TokenizationFunctions.CreatePostInputSplitter(ruleset.EOL));

			foreach (Token token in tokenizer.Tokenize(content))
			{
				Console.WriteLine($"{token.Value} - {token.Type}");
			}

			var tokens = tokenizer.Tokenize(content);
			Console.WriteLine(string.Join(" ", tokens.Select(token => $"{token.Value}")));
		}
	}
}