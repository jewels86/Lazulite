using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Threading.Tasks;

namespace Lazulite.Tokenization 
{
	public delegate IEnumerable<PartialToken> InputSplitterDelegate(string input);
	public delegate IEnumerable<PartialToken> PostInputSplitterDelegate(IEnumerable<PartialToken> parts);

	public interface IInputSplitting 
	{
		public void SetInputSplitter(InputSplitterDelegate inputSplitter);
		public void SetPostInputSplitter(PostInputSplitterDelegate postInputSplitter);
	}
}