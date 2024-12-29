using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Tokenization
{
	public class PartialToken
	{
		public string Value { get; }
		public int Length => Value.Length;
		public int StartIndex { get; }

		public PartialToken(string value, int startIndex)
		{
			Value = value;
			StartIndex = startIndex;
		}
	}
}
