using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Tokenization
{
	public class Token
	{
		public int StartIndex { get; }
		public int EndIndex { get; }
		public string Value { get; }
		public int Length => Value.Length;
		public Dictionary<string, object> Metadata { get; } = [];
		public string Type { get; }

		public Token(int startIndex, string value, string type)
		{
			StartIndex = startIndex;
			EndIndex = startIndex + value.Length;
			Value = value;
			Type = type;
		}
	}
}
