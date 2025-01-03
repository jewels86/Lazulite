using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing
{
	public class ParserContext
	{
		private List<Token> _tokens;
		private int _index;

		public ParserContext(List<Token> tokens)
		{
			_tokens = tokens;
			_index = 0;
		}

		public Token Current() => _tokens[_index];
		public void Consume() => _index++;
		public int Index => _index;
		public void Restore(int index) => _index = index;

	}
}
