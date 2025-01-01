using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Lazulite.Parsing
{
    public interface IAstNode
    {
        public string Type { get; set; }
    }

    public class LiteralAstNode : IAstNode
    {
        public string Value { get; set; }
        public string Type { get; set; }

        public LiteralAstNode(string value, string type)
        {
            Value = value;
            Type = type;
        }
    }
}