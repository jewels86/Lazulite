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
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromList(["int", "string", "char"], "type"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromList(["=", "+", "-", "*", "/"], "operator"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"\d+", "number"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"[a-zA-Z_]\w*", "identifier"));
			foreach (Token token in tokenizer.Tokenize(content))
			{
				Console.WriteLine($"{token.Value} - {token.Type}");
			}
			var tokens = tokenizer.Tokenize(content);
			Console.WriteLine(string.Join(" ", tokens.Select(token => $"{token.Value}")));
		}
	}
}