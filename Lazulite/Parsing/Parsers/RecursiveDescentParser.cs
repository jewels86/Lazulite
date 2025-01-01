using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing.Parsers
{
    public class RecursiveDescentParser 
    {
        private readonly List<GrammarRuleDelegate> _rules = [];
        private ParserErrorDelegate _errorHandler = (ctx) => throw new Exception();

        public void AddRule(GrammarRuleDelegate rule)
        {
            _rules.Add(rule);
        }
        public void AddRules(IEnumerable<GrammarRuleDelegate> rules)
        {
            _rules.AddRange(rules);
        }

        public IAstNode? Parse(ParserContext context) 
        {
            foreach (GrammarRuleDelegate rule in _rules)
            {
                if (rule(context, out IAstNode? node)) return node!;
            }
            _errorHandler(context);
            return null;
        }
    }
}