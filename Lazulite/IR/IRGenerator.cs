using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.IR
{
	public delegate List<IRInstruction> IRCreator<T>(string code, T context);
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
	
		public void Generate(string type, string code)
		{

		}
	}
}
