using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public class IRInstruction<TInstruction, TOperand> where TInstruction : Enum where TOperand : Enum
	{
		public TInstruction Instruction { get; set; }
		public List<TOperand> Operands { get; set; }

		public IRInstruction(TInstruction instruction, List<TOperand> operands)
		{
			Instruction = instruction;
			Operands = operands;
		}

		public override string ToString()
		{
			return $"IRInstruction({Instruction}, {string.Join(", ", Operands)})";
		}
	}
}
