using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Parsing;

namespace Lazulite.IR.Utilities
{
	public class AsmParser : IParser<string>
	{
		public IAstNode? Parse(ParserContext<string> ctx)
		{
			throw new NotImplementedException(); // will be implemented with LPN
		}
	}
}
