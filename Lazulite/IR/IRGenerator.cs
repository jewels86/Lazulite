using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Parsing;

namespace Lazulite.IR
{
	public delegate List<IRInstruction> IRCreator<T>(IAstNode ast, T context);
	public delegate void IRModifier<T>(out List<IRInstruction> instructions, T context);

	public class IRGenerator<T>
	{
		public T Context { get; set; }
		public List<IRInstruction> Instructions { get; set; } = [];

		public Dictionary<string, IRCreator<T>> Creators = [];
		public Dictionary<string, List<IRModifier<T>>> Modifiers = [];

		public IRGenerator(T context)
		{
			Context = context;
		}

		public void SetCreator(string code, IRCreator<T> creator)
		{
			Creators[code] = creator;
		}
		public void AddModifier(string codeType, IRModifier<T> modifier)
		{
			if (!Modifiers.ContainsKey(codeType))
			{
				Modifiers[codeType] = [];
			}
			Modifiers[codeType].Add(modifier);
		}

		public List<IRInstruction> Generate(string type, IAstNode ast)
		{
			if (Creators.TryGetValue(type, out var creator))
			{
				var instructions = creator(ast, Context);
				if (Modifiers.TryGetValue(type, out var modifiers))
				{
					foreach (var modifier in modifiers) modifier(out instructions, Context);
				}
				return instructions;
			}
			else throw new Exception($"No creator found for type {type}");
		}
	}
}
