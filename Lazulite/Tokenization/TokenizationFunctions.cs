using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lazulite.Tokenization
{
	public static class TokenizationFunctions
	{
		public static TokenRuleDelegate CreateRuleFromList(List<string> matches, string type)
		{
			return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
			{
				if (matches.Contains(part.Value))
				{
					token = new Token(part.StartIndex, part.Value, type);
					return true;
				}
				token = null;
				return false;
			};
		}
		public static TokenRuleDelegate CreateRuleFromPredicate(Func<string, bool> predicate, string type)
		{
			return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
			{
				if (predicate(part.Value))
				{
					token = new Token(part.StartIndex, part.Value, type);
					return true;
				}
				token = null;
				return false;
			};
		}
		public static TokenRuleDelegate CreateRuleFromRegex(string pattern, string type)
		{
			return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
			{
				if (System.Text.RegularExpressions.Regex.IsMatch(part.Value, pattern))
				{
					token = new Token(part.StartIndex, part.Value, type);
					return true;
				}
				token = null;
				return false;
			};
		}

		public static IEnumerable<PartialToken> DefaultInputSplitter(string input)
		{
			return SeperatorInputSplitter(input, " ", "\n", "\t", "\r");
		}
		public static IEnumerable<PartialToken> SeperatorInputSplitter(string input, params string[] separators)
		{
			int start = 0;
			while (start < input.Length)
			{
				int minIndex = -1;
				string minSeparator = string.Empty;

				foreach (var separator in separators)
				{
					int index = input.IndexOf(separator, start);
					if (index != -1 && (minIndex == -1 || index < minIndex))
					{
						minIndex = index;
						minSeparator = separator;
					}
				}

				if (minIndex == -1)
				{
					yield return new PartialToken(input.Substring(start), start);
					yield break;
				}

				if (minIndex > start)
				{
					yield return new PartialToken(input.Substring(start, minIndex - start), start);
				}

				start = minSeparator != null ? minIndex + minSeparator.Length : minIndex;
			}
		}

		public static PostInputSplitterDelegate CreatePostInputSplitter(string? eol = null)
		{
			IEnumerable<PartialToken> Function(IEnumerable<PartialToken> input)
			{
				input = CleanSpiltInput(input);
				foreach (PartialToken ptoken in input)
				{
					if (eol is not null && ptoken.Value.Contains(eol))
					{
						int startIndex = ptoken.StartIndex;
						string[] parts = ptoken.Value.Split(new[] { eol }, StringSplitOptions.None);
						for (int i = 0; i < parts.Length; i++)
						{
							if (i > 0)
							{
								yield return new PartialToken(eol, startIndex);
								startIndex += eol.Length;
							}
							if (!string.IsNullOrEmpty(parts[i]))
							{
								yield return new PartialToken(parts[i], startIndex);
								startIndex += parts[i].Length;
							}
						}
					}
					else
					{
						yield return ptoken;
					}
				}
			}
			return Function;
		}

		public static IEnumerable<PartialToken> CleanSpiltInput(IEnumerable<PartialToken> input)
		{
			return input.Where(token => !string.IsNullOrWhiteSpace(token.Value));
		}

		public static TokenRuleDelegate CreateMatchAllRule(string type)
		{
			return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
			{
				token = new Token(part.StartIndex, part.Value, type);
				return true;
			};
		}
	}
}
