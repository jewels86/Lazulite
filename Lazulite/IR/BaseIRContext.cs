using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public class BaseIRContext
	{
		public List<IRInstruction> Instructions { get; set; } = [];
		public List<IROperand> Variables { get; set; } = [];
		public List<IROperand> Constants { get; set; } = [];
		public List<IROperand> Types { get; set; } = [];
		public List<IROperand> Methods { get; set; } = [];
	}
}
