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
		public Dictionary<string, string> Types { get; set; } = new();
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
		public void AddType(string match, string type)
		{
			Types.Add(type, match);
		}
		public void AddType(string match)
		{
			Types.Add("type", match);
		}
		public void AddTypes(string[] matches)
		{
			foreach (var match in matches)
			{
				AddType(match);
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

		public IEnumerable<TokenRuleDelegate> Unpack()
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
	}
}
