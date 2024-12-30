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

			StandardRuleset ruleset = new(";", "//");
			ruleset.AddType("int", "type");
			ruleset.AddTypeLiteral("int-literal", @"\d+");
			ruleset.AddOperator("+");
			ruleset.AddOperator("=");
			ruleset.SetIdentifier(@"[a-zA-Z_]\w*");
			tokenizer.AddRules(ruleset.Unpack());

			foreach (Token token in tokenizer.Tokenize(content))
			{
				Console.WriteLine($"{token.Value} - {token.Type}");
			}

			var tokens = tokenizer.Tokenize(content);
			Console.WriteLine(string.Join(" ", tokens.Select(token => $"{token.Value}")));
		}
	}
}