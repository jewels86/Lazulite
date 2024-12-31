using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Tokenization.Tokenizers
{
	public class StandardTokenizer : ITokenizer, IStringTokenizer
	{
		private readonly List<TokenRuleDelegate> _rules = [];
		private readonly List<PostProcessorDelegate> _postProcessors = [];
		private TokenizerErrorDelegate _errorHandler = (input, index) => throw new Exception($"No token type applied for index {index} ({input[index]})");

		public void AddRule(TokenRuleDelegate rule)
		{
			_rules.Add(rule);
		}
		public void AddRules(IEnumerable<TokenRuleDelegate> rules)
		{
			_rules.AddRange(rules);
		}

		public void AddPostProcessor(PostProcessorDelegate postProcessor)
		{
			_postProcessors.Add(postProcessor);
		}
		public void AddPostProcessors(IEnumerable<PostProcessorDelegate> postProcessors)
		{
			_postProcessors.AddRange(postProcessors);
		}

		public void SetErrorHandler(TokenizerErrorDelegate tokenizerErrorDelegate)
		{
			_errorHandler = tokenizerErrorDelegate;
		}

		public IEnumerable<Token> Tokenize(string input)
		{
			List<Token> tokens = new List<Token>();
			int index = 0;
			while (index < input.Length)
			{
				Token? token = null;
				foreach (TokenRuleDelegate rule in _rules)
				{
					if (rule(input, index, out token))
					{
						tokens.Add(token!);
						//Console.WriteLine($"{token!.Value} - {token!.Type} ({index}-{index + token!.Length})");
						index += token!.Length;
						break;
					}
				}
				if (token == null)
				{
					_errorHandler(input, index);
					break;
				}
			}

			foreach (PostProcessorDelegate postProcessor in _postProcessors)
			{
				tokens = postProcessor(tokens).ToList();
			}

			return tokens;
		}
	}
}
