using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing
{
	public class ParserContext<T>
	{
		private List<T> _list;
		private int _index;

		public ParserContext(List<T> list)
		{
			_list = list;
			_index = 0;
		}

		public T Current() => _list[_index];
		public void Consume() => _index++;
		public int Index => _index;
		public void Restore(int index) => _index = index;

	}
}
