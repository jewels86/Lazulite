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
        public string NodeType { get; }
    }
}