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
			ruleset.AddTypes(["int"]);
			ruleset.AddTypeLiteral("int", @"[0-9]+");
			ruleset.AddTypeLiteral("bool", @"true|false");
			tokenizer.AddRules(ruleset.Unpack());

			foreach (Token token in tokenizer.Tokenize(content))
			{
				Console.WriteLine($"{token.Value} - {token.Type}");
			}

			var tokens = tokenizer.Tokenize(content);
			Console.WriteLine(string.Join("", tokens.Select(token => $"{token.Value}")));
		}
	}
}