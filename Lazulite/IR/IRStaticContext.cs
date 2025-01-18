using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public class IRStaticContext<TInstruction, TOperand> where TInstruction : Enum where TOperand : Enum
	{
		public List<IRInstruction<TInstruction, TOperand>> Instructions { get; set; } = [];

		public IRInstruction<TInstruction, TOperand> this[int index]
		{
			get => Instructions[index];
			set => Instructions[index] = value;
		}
	}
}
