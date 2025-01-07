using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing.Lpn
{
	public class LpnProcessor<T>
	{
		public List<IGrammarRule<T>> Process(string input, LpnContext<T> contex)
		{
			StandardTokenizer tokenizer = new();
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"[a-zA-Z_][a-zA-Z0-9_-]*", "identifier"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(TokenizationFunctions.StandardIntegerLiteralRegex, "integer"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(TokenizationFunctions.StandardStringLiteralRegex, "string"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(TokenizationFunctions.StandardFloatLiteralRegex, "float"));
			tokenizer.AddRules(TokenizationFunctions.CreateBraceRules());
			tokenizer.AddRules(TokenizationFunctions.CreateParenthesisRules());
			tokenizer.AddRules(TokenizationFunctions.CreateBracketRules());
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"=>", "function-operator"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"\.", "member-operator"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"=", "declaration-operator"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"@", "metadata-operator"));
			tokenizer.AddRules(TokenizationFunctions.CreateWhitespaceRules());
			tokenizer.AddPostProcessor(TokenizationFunctions.CreatePostProcessorFilterForWhitespaces());

			var tokens = tokenizer.Tokenize(input);


			return [];
		}
	}
}