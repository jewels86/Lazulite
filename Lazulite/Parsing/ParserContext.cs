using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing
{
	public class ParserContext
	{
		public List<Token> Tokens { get; set;}
		public int Index { get; set;}

		public ParserContext(IEnumerable<Token> tokens)
		{
			Tokens = tokens.ToList();
			Index = 0;
		}

		public Token CurrentToken() => Tokens[Index];
		public void Consume() => Index += 1;
		public bool HasNext() => Index < Tokens.Count;
	}
}