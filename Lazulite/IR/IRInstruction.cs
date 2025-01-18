using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public class IRInstruction<T> where T : Enum
	{
		public T Instruction { get; set; }
		public List<IIROperand> Operands { get; set; } = new List<IIROperand>();

		public IRInstruction(T instruction, IIROperand[] operands)
		{
			Instruction = instruction;
			Operands.AddRange(operands);
		}

		public void AddOperand(IIROperand operand)
		{
			Operands.Add(operand);
		}
		public void AddOperands(IIROperand[] operands)
		{
			Operands.AddRange(operands);
		}
		public void InsertOperand(int index, IIROperand operand)
		{
			Operands.Insert(index, operand);
		}
		public void InsertOperands(int index, IIROperand[] operands)
		{
			Operands.InsertRange(index, operands);
		}

		public IIROperand this[int index]
		{
			get => Operands[index];
			set => Operands[index] = value;
		}
	}
}
