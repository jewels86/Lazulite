using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Lazulite.Tokenization
{
	public class Ruleset
	{
		public string EOL { get; set; }
		public string SingleLineComment { get; set; }
		public string? BlockComment { get; set; }
		public List<KeyValuePair<string, string>> Types { get; set; } = new();
		public Dictionary<string, Regex> TypeLiterals { get; set; } = new();
		public Regex? Identifier { get; set; }
		public List<string> Operators { get; set; } = new();

		public Ruleset(string endOfLine, string singleLineComment)
		{
			EOL = endOfLine;
			SingleLineComment = singleLineComment;
		}

		public void AddTypeLiteral(string type, string regex)
		{
			TypeLiterals.Add(type, new Regex(regex));
		}
		public void AddType(string match, string type = "type")
		{
			Types.Add(new(type, match));
		}
		public void AddTypes(string[] matches, string type = "type")
		{
			foreach (var match in matches)
			{
				AddType(match, type);
			}
		}
		public void AddOperator(string op)
		{
			if (!Operators.Contains(op))
			{
				Operators.Add(op);
			}
		}
		public void AddOperators(string[] ops)
		{
			foreach (var op in ops)
			{
				AddOperator(op);
			}
		}
		public void SetIdentifier(string regex)
		{
			Identifier = new Regex(regex);
		}

		public IEnumerable<SplitInputTokenRuleDelegate> UnpackForSplitInput()
		{
			yield return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
			{
				if (part.Value == EOL)
				{
					token = new Token(part.StartIndex, part.Value, "end-of-line");
					return true;
				}
				token = null;
				return false;
			};

			yield return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
			{
				if (part.Value == SingleLineComment)
				{
					token = new Token(part.StartIndex, part.Value, "single-line-comment");
					return true;
				}
				token = null;
				return false;
			};

			if (BlockComment != null)
			{
				yield return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
				{
					if (part.Value == BlockComment)
					{
						token = new Token(part.StartIndex, part.Value, "block-comment");
						return true;
					}
					token = null;
					return false;
				};
			}

			foreach (var type in Types)
			{
				yield return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
				{
					if (part.Value == type.Value)
					{
						token = new Token(part.StartIndex, part.Value, type.Key);
						return true;
					}
					token = null;
					return false;
				};
			}
			foreach (var typeLiteral in TypeLiterals)
			{
				yield return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
				{
					if (typeLiteral.Value.IsMatch(part.Value))
					{
						token = new Token(part.StartIndex, part.Value, typeLiteral.Key);
						return true;
					}
					token = null;
					return false;
				};
			}

			if (Identifier is not null)
			{
				yield return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
				{
					if (Identifier.IsMatch(part.Value))
					{
						token = new Token(part.StartIndex, part.Value, "identifier");
						return true;
					}
					token = null;
					return false;
				};
			}

			foreach (var op in Operators)
			{
				yield return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
				{
					if (part.Value == op)
					{
						token = new Token(part.StartIndex, part.Value, "operator");
						return true;
					}
					token = null;
					return false;
				};
			}
		}
		public IEnumerable<TokenRuleDelegate> Unpack()
		{
			yield return (string input, int index, out Token? token) =>
			{
				token = null;
				if (input.Substring(index).StartsWith(EOL))
				{
					token = new Token(index, EOL, "end-of-line");
					return true;
				}
				return false;
			};
			yield return (string input, int index, out Token? token) =>
			{
				token = null;
				if (input.Substring(index).StartsWith(SingleLineComment))
				{
					for (int i = index; i < input.Length; i++)
					{
						if (input[i] == '\n')
						{
							token = new Token(index, input.Substring(index, i - index), "single-line-comment");
							return true;
						}
					}
				}
				return false;
			};

			if (BlockComment != null) yield return (string input, int index, out Token? token) =>
			{
				token = null;
				string sub = input.Substring(index);
				if (sub.StartsWith(BlockComment))
				{
					string block = sub.Substring(BlockComment.Length);
					int end = block.IndexOf(BlockComment);
					if (end != -1)
					{
						token = new Token(index, block.Substring(0, end), "block-comment");
						return true;
					}
				}
				return false;
			};

			foreach (var type in Types)
			{
				yield return (string input, int index, out Token? token) =>
				{
					token = null;
					if (input.Substring(index).StartsWith(type.Value))
					{
						token = new Token(index, type.Value, type.Key);
						return true;
					}
					return false;
				};
			}
			foreach (var typeLiteral in TypeLiterals)
			{
				yield return (string input, int index, out Token? token) =>
				{
					token = null;
					string sub = input.Substring(index);
					if (typeLiteral.Value.IsMatch(sub))
					{
						token = new Token(index, typeLiteral.Value.Match(sub).Value, typeLiteral.Key);
						return true;
					}
					return false;
				};
			}
			foreach (var op in Operators)
			{
				yield return (string input, int index, out Token? token) =>
				{
					token = null;
					if (input.Substring(index).StartsWith(op))
					{
						token = new Token(index, op, "operator");
						return true;
					}
					return false;
				};
			}

			if (Identifier != null) yield return (string input, int index, out Token? token) =>
			{
				token = null;
				string sub = input.Substring(index);
				if (Identifier.IsMatch(sub[0].ToString()))
				{
					int length = 1;
					while (length < sub.Length && Identifier.IsMatch(sub[length].ToString()))
					{
						length++;
					}
					token = new Token(index, sub.Substring(0, length), "identifier");
					return true;
				}
				return false;
			};
		}
	}
}