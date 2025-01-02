using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Parsing 
{
    public static class ParsingFunctions 
    {
        public static void CreateParsingRuleFromEBNF(string ebnf, ref Dictionary<string, GrammarRuleDelegate> table) 
        {
            string ruleName = ebnf.Split('=')[0].Trim();
            
        }
    }
}