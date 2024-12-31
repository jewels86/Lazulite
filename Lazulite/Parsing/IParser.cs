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
    public interface IParser
    {


        public AstNode Parse(IEnumerable<Token> tokens);
    }
}