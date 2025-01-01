using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Lazulite.Tokenization;

namespace Lazulite.Parsing
{
    public delegate bool GrammarRuleDelegate(ParserContext context, out IAstNode? node);
    public delegate void ParserErrorDelegate(ParserContext context);

    public interface IParser
    {
        public IAstNode? Parse(ParserContext context);
    }
}