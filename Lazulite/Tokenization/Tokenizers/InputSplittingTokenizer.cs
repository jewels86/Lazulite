using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Lazulite.Tokenization.TokenizationFunctions;

namespace Lazulite.Tokenization.Tokenizers
{
	public class InputSplittingTokenizer : ITokenizer, IInputSplitting
	{
		private readonly List<SplitInputTokenRuleDelegate> _rules = [];
		private readonly List<PostProcessorDelegate> _postProcessors = [];
		private InputSplitterDelegate _inputSplitter = DefaultInputSplitter;
		private PostInputSplitterDelegate? _postInputSplitter;
		private TokenizerErrorDelegate _errorHandler = (input, index) => throw new Exception($"No token type applied for index {index} ({input[index]})");

		public void AddRule(SplitInputTokenRuleDelegate rule)
		{
			_rules.Add(rule);
		}
		public void AddRules(IEnumerable<SplitInputTokenRuleDelegate> rules)
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

		public void SetInputSplitter(InputSplitterDelegate inputSplitter)
		{
			_inputSplitter = inputSplitter;
		}
		public void SetPostInputSplitter(PostInputSplitterDelegate postInputSplitter)
		{
			_postInputSplitter = postInputSplitter;
		}
		public void SetErrorHandler(TokenizerErrorDelegate tokenizerErrorDelegate)
		{
			_errorHandler = tokenizerErrorDelegate;
		}

		public IEnumerable<Token> Tokenize(string input)
		{
			IEnumerable<PartialToken> parts = _inputSplitter(input);
			IEnumerable<Token> tokens = new List<Token>();
			parts = _postInputSplitter != null ? _postInputSplitter(parts) : parts;

			int index = 0;
			foreach (PartialToken part in parts)
			{
				Token? token = null;
				foreach (SplitInputTokenRuleDelegate rule in _rules)
				{
					if (rule(parts, part, index, out token))
					{
						tokens.Append(token!);
						break;
					}
				}
				if (token == null)
				{
					_errorHandler(input, part.StartIndex);
				}
			}
			foreach (PostProcessorDelegate postProcessor in _postProcessors)
			{
				tokens = postProcessor(tokens);
			}
			return tokens;
		}
	}
}
