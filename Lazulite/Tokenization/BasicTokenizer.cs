using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Lazulite.Tokenization.TokenizationFunctions;

namespace Lazulite.Tokenization
{
	public class BasicTokenizer: ITokenizer, IInputSplitting
	{
		private readonly List<TokenRuleDelegate> _rules = [];
		private readonly List<TokenPostProcessorDelegate> _postProcessors = [];
		private InputSplitterDelegate _inputSplitter = DefaultInputSplitter;
		private TokenizerErrorDelegate _errorHandler = (string message) => throw new Exception(message);

		public void AddRule(TokenRuleDelegate rule)
		{
			_rules.Add(rule);
		}
		public void AddRules(IEnumerable<TokenRuleDelegate> rules)
		{
			_rules.AddRange(rules);
		}

		public void AddPostProcessor(TokenPostProcessorDelegate postProcessor)
		{
			_postProcessors.Add(postProcessor);
		}
		public void AddPostProcessors(IEnumerable<TokenPostProcessorDelegate> postProcessors)
		{
			_postProcessors.AddRange(postProcessors);
		}

		public void SetInputSplitter(InputSplitterDelegate inputSplitter)
		{
			_inputSplitter = inputSplitter;
		}
		public void SetErrorHandler(TokenizerErrorDelegate tokenizerErrorDelegate)
		{
			_errorHandler = tokenizerErrorDelegate;
		}

		public IEnumerable<Token> Tokenize(string input)
		{
			IEnumerable<PartialToken> parts = _inputSplitter(input);

			int index = 0;
			foreach (PartialToken part in parts)
			{
				Token? token = null;
				foreach (TokenRuleDelegate rule in _rules)
				{
					if (rule(parts, part, index, out token))
					{
						foreach (TokenPostProcessorDelegate postProcessor in _postProcessors)
						{
							token = postProcessor(token!);
						}
						yield return token!;
						break;
					}
				}
				if (token == null)
				{
					_errorHandler($"No rule matched for token '{part.Value}' at index {part.StartIndex}");
				}
			}
		}
	}
}
