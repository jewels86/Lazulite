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
			BasicTokenizer tokenizer = new();

			Ruleset ruleset = new(";", "//");
			ruleset.AddTypes(["int", "string", "char", "float"]);
			ruleset.AddTypeLiteral("int-literal", @"\d+");
			ruleset.AddOperators(["+", "-", "*", "/", "="]);
			ruleset.SetIdentifier(TokenizationFunctions.StandardIdentifierRegex);
			tokenizer.AddRules(ruleset.Unpack());
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