using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public abstract class BaseIRContext<T> where T : Enum
	{
		public List<IRInstruction<T>> Instructions { get; set; } = [];
		public List<IIROperand> Variables { get; set; } = [];
		public List<IIROperand> Methods { get; set; } = [];
		public List<IIROperand> Types { get; set; } = [];

		#region Adding/Inserting Methods
		public void AddInstruction(IRInstruction<T> instruction)
		{
			Instructions.Add(instruction);
		}
		public void AddVariable(IIROperand variable)
		{
			Variables.Add(variable);
		}
		public void AddMethod(IIROperand method)
		{
			Methods.Add(method);
		}
		public void AddType(IIROperand type)
		{
			Types.Add(type);
		}

		public void AddInstructions(IRInstruction<T>[] instructions)
		{
			Instructions.AddRange(instructions);
		}
		public void AddVariables(IIROperand[] variables)
		{
			Variables.AddRange(variables);
		}
		public void AddMethods(IIROperand[] methods)
		{
			Methods.AddRange(methods);
		}
		public void AddTypes(IIROperand[] types)
		{
			Types.AddRange(types);
		}

		public void InsertInstruction(int index, IRInstruction<T> instruction)
		{
			Instructions.Insert(index, instruction);
		}
		public void InsertVariable(int index, IIROperand variable)
		{
			Variables.Insert(index, variable);
		}
		public void InsertMethod(int index, IIROperand method)
		{
			Methods.Insert(index, method);
		}
		public void InsertType(int index, IIROperand type)
		{
			Types.Insert(index, type);
		}

		public void InsertInstructions(int index, IRInstruction<T>[] instructions)
		{
			Instructions.InsertRange(index, instructions);
		}
		public void InsertVariables(int index, IIROperand[] variables)
		{
			Variables.InsertRange(index, variables);
		}
		public void InsertMethods(int index, IIROperand[] methods)
		{
			Methods.InsertRange(index, methods);
		}
		public void InsertTypes(int index, IIROperand[] types)
		{
			Types.InsertRange(index, types);
		}
		#endregion

		public IRInstruction<T> this[int index]
		{
			get => Instructions[index];
			set => Instructions[index] = value;
		}
	}
}