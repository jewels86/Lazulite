using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Threading.Tasks;

namespace Lazulite.Tokenization 
{
    public interface IInputSplitting 
    {
        public void SetInputSplitter(InputSplitterDelegate inputSplitter);
    }
}