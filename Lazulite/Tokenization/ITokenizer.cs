using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Tokenization
{
	public delegate bool TokenRuleDelegate(IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token);
	public delegate Token TokenPostProcessorDelegate(Token token);
	public delegate IEnumerable<PartialToken> InputSplitterDelegate(string input);
	public delegate void TokenizerErrorDelegate(string message);

	internal interface ITokenizer
	{
		public void AddRule(TokenRuleDelegate rule);
		public void AddRules(IEnumerable<TokenRuleDelegate> rules);
		public void AddPostProcessor(TokenPostProcessorDelegate postProcessor);
		public void AddPostProcessors(IEnumerable<TokenPostProcessorDelegate> postProcessors);
		public void SetErrorHandler(TokenizerErrorDelegate errorHandler);
		public IEnumerable<Token> Tokenize(string input);
	}
}
