using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing.Lpn
{
	public class LpnContext<T>
	{
		public Dictionary<string, IGrammarRule<T>> Rules { get; private set; } = [];
	}
}
