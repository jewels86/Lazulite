using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public class IRInstruction
	{
		public string Instruction { get; set; }
		public List<IROperand> Operands { get; set; }

		public IRInstruction(string instruction, List<IROperand> operands)
		{
			Instruction = instruction;
			Operands = operands;
		}
		public IRInstruction(string instruction, params IROperand[] operands)
		{
			Instruction = instruction;
			Operands = operands.ToList();
		}
		public IRInstruction(string instruction)
		{
			Instruction = instruction;
			Operands = [];
		}

		public void AddOperand(IROperand operand)
		{
			Operands.Add(operand);
		}
		public void AddOperands(IROperand[] operands)
		{
			Operands.AddRange(operands);
		}
		public void InsertOperand(int index, IROperand operand)
		{
			Operands.Insert(index, operand);
		}
		public void InsertOperands(int index, IROperand[] operands)
		{
			Operands.InsertRange(index, operands);
		}

		public IROperand this[int index]
		{
			get => Operands[index];
			set => Operands[index] = value;
		}
	}
}
