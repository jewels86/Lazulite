using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Tokenization
{
	public delegate bool SplitInputTokenRuleDelegate(IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token);
	public delegate IEnumerable<PartialToken> InputSplitterDelegate(string input);
	public delegate IEnumerable<PartialToken> PostInputSplitterDelegate(IEnumerable<PartialToken> parts);
	public delegate bool TokenRuleDelegate(string input, int index, out Token? token);
	public delegate IEnumerable<Token> PostProcessorDelegate(IEnumerable<Token> tokens);
	public delegate void TokenizerErrorDelegate(string input, int index);

	internal interface ITokenizer
	{
		public void AddPostProcessor(PostProcessorDelegate postProcessor);
		public void AddPostProcessors(IEnumerable<PostProcessorDelegate> postProcessors);
		public void SetErrorHandler(TokenizerErrorDelegate errorHandler);
		public IEnumerable<Token> Tokenize(string input);
	}
	internal interface IInputSplitting
	{
		public void AddRule(SplitInputTokenRuleDelegate rule);
		public void AddRules(IEnumerable<SplitInputTokenRuleDelegate> rules);
		public void SetInputSplitter(InputSplitterDelegate inputSplitter);
		public void SetPostInputSplitter(PostInputSplitterDelegate postInputSplitter);
	}
	internal interface IStringTokenizer
	{
		public void AddRule(TokenRuleDelegate rule);
		public void AddRules(IEnumerable<TokenRuleDelegate> rules);
	}
}
