using Lazulite.IR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing.Lpn
{
	public class LpnContext<T>
	{
		public Dictionary<string, IGrammarRule<T>> Rules { get; private set; } = [];
		public Dictionary<string, List<string>> Metadata { get; private set; } = [];

		public void AddMetadata(string key, string[] values)
		{
			if (!Metadata.ContainsKey(key)) Metadata[key] = [];
			Metadata[key].AddRange(values);
		}
	}

	public class LpnRule<T>
	{
		public IGrammarRule<T> Rule { get; private set; }
		public GrammarRuleActionDelegate Action { get; private set; }

		public BlockContext Block { get; private set; } = new();
		
		public LpnRule(IGrammarRule<T> rule, GrammarRuleActionDelegate action)
		{
			Rule = rule;
			Action = action;
		}
	}

	public class BlockContext
	{
		public Dictionary<string, IROperand> Variables { get; private set; } = [];
		public Dictionary<string, IROperand> Constants { get; private set; } = [];
		public Dictionary<string, IROperand> Types { get; private set; } = [];
		public Dictionary<string, IROperand> Methods { get; private set; } = [];
	}
}
