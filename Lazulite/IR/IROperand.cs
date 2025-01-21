using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public class IROperand
	{
		public string Type { get; set; }
		public object Value { get; set; }

		public IROperand(string type, object value)
		{
			Type = type;
			Value = value;
		}
	}
}
