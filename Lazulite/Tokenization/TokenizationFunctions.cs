﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace Lazulite.Tokenization
{
	public static class TokenizationFunctions
	{
		#region Standard Tokenization Regexes & Lists
		public static string StandardIdentifierRegex = @"[a-zA-Z_][a-zA-Z0-9_]*";
		public static string StandardIntegerLiteralRegex = @"\d+";
		public static string StandardFloatLiteralRegex = @"\d+\.\d+";
		public static string StandardStringLiteralRegex = @"""[^""]*""";
		public static string StandardCharLiteralRegex = @"'\S'";
		public static string StandardLowercaseBooleanLiteralRegex = @"true|false";
		public static string StandardUppercaseBooleanLiteralRegex = @"True|False";
		public static string[] StandardMathOperators = ["+", "-", "/", "*"];
		public static string[] StandardComparisonOperators = ["==", "!=", "<", ">", "<=", ">="];
		public static string[] StandardLogicalOperators = ["&&", "||", "!"];
		public static string[] StandardAssignmentOperators = ["=", "+=", "-=", "*=", "/="];
		public static string[] StandardKeywords = ["if", "else", "for"];
		#endregion

		#region Token Rules
		public static TokenRuleDelegate CreateRuleFromString(string match, string type)
		{
			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				if (input.Substring(index).StartsWith(match))
				{
					token = new Token(index, match, type);
					return true;
				}

				return false;
			};
		}
		public static TokenRuleDelegate CreateRuleFromList(List<string> matches, string type)
		{
			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				ReadOnlySpan<char> span = input.AsSpan(index);

				foreach (string match in matches)
				{
					if (span.StartsWith(match))
					{
						token = new Token(index, match, type);
						return true;
					}
				}
				return false;
			};
		}
		public static TokenRuleDelegate CreateRuleFromPredicate(Func<string, bool> predicate, string type)
		{
			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				ReadOnlySpan<char> span = input.AsSpan(index);

				for (int i = 0; i < span.Length; i++)
				{
					var sub = span.Slice(0, i + 1);
					if (predicate(sub.ToString()))
					{
						token = new Token(index, sub.ToString(), type);
						return true;
					}
				}

				return false;
			};
		}
		public static TokenRuleDelegate CreateRuleFromRegex(string pattern, string type)
		{
			Regex regex = new Regex(pattern);

			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				ReadOnlySpan<char> span = input.AsSpan(index);

				for (int i = 0; i < span.Length; i++)
				{
					var sub = span.Slice(0, i + 1);
					if (regex.IsMatch(sub))
					{
						token = new Token(index, sub.ToString(), type);
						return true;
					}
				}

				return false;
			};
		}
		public static TokenRuleDelegate CreateSpaceRule(string type = "space")
		{
			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				if (input[index] == ' ')
				{
					token = new Token(index, " ", type);
					return true;
				}

				return false;
			};
		}
		public static TokenRuleDelegate CreateNewlineRule(string type = "newline")
		{
			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				if (input[index] == '\n')
				{
					token = new Token(index, "\n", type);
					return true;
				}

				return false;
			};
		}
		public static TokenRuleDelegate CreateTabRule(string type = "tab")
		{
			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				if (input[index] == '\t')
				{
					token = new Token(index, "\t", type);
					return true;
				}

				return false;
			};
		}
		public static TokenRuleDelegate CreateReturnRule(string type = "return")
		{
			return (string input, int index, out Token? token) =>
			{
				token = null;
				if (index >= input.Length) return false;

				if (input[index] == '\r')
				{
					token = new Token(index, "\r", type);
					return true;
				}

				return false;
			};
		}
		public static IEnumerable<TokenRuleDelegate> CreateWhitespaceRules()
		{
			yield return CreateSpaceRule();
			yield return CreateNewlineRule();
			yield return CreateTabRule();
			yield return CreateReturnRule();
		}
		public static IEnumerable<TokenRuleDelegate> CreateParenthesisRules()
		{
			yield return CreateRuleFromString("(", "left-parenthesis");
			yield return CreateRuleFromString(")", "right-parenthesis");
		}
		public static IEnumerable<TokenRuleDelegate> CreateBracketRules()
		{
			yield return CreateRuleFromString("[", "left-bracket");
			yield return CreateRuleFromString("]", "right-bracket");
		}
		public static IEnumerable<TokenRuleDelegate> CreateBraceRules()
		{
			yield return CreateRuleFromString("{", "left-brace");
			yield return CreateRuleFromString("}", "right-brace");
		}
		public static IEnumerable<TokenRuleDelegate> CreateAngleBracketRules()
		{
			yield return CreateRuleFromString("<", "left-angle-bracket");
			yield return CreateRuleFromString(">", "right-angle-bracket");
		}
		public static IEnumerable<TokenRuleDelegate> CreateQuoteRules()
		{
			yield return CreateRuleFromString("\"", "quote");
			yield return CreateRuleFromString("'", "single-quote");
		}
		public static TokenRuleDelegate CreateSemicolonRule() 
		{
			return CreateRuleFromString(";", "semicolon");
		}
		public static TokenRuleDelegate CreateCommaRule()
		{
			return CreateRuleFromString(",", "comma");
		}
		public static TokenRuleDelegate CreateMatchAllRule(string type = "unidentified")
		{
			return (string input, int index, out Token? token) =>
			{
				token = new Token(index, input[index].ToString(), type);
				return true;
			};
		}
		#endregion

		#region Post Processors
		public static PostProcessorDelegate CreatePostProcessorFilterFromType(string type, bool allow)
		{
			return (IEnumerable<Token> tokens) =>
			{
				List<Token> filtered = new List<Token>();
				foreach (Token token in tokens)
				{
					if (token.Type == type)
					{
						if (allow)
						{
							filtered.Add(token);
						}
					}
					else
					{
						if (!allow)
						{
							filtered.Add(token);
						}
					}
				}
				return filtered;
			};
		}
		public static PostProcessorDelegate CreatePostProcessorFilterFromPredicate(Func<Token, bool> predicate, bool allow)
		{
			return (IEnumerable<Token> tokens) =>
			{
				IEnumerable<Token> filtered = new List<Token>();
				foreach (Token token in tokens)
				{
					if (predicate(token))
					{
						if (allow)
						{
							filtered.Append(token);
						}
					}
					else
					{
						if (!allow)
						{
							filtered.Append(token);
						}
					}
				}
				return filtered;
			};
		}
		public static PostProcessorDelegate CreatePostProcessorFilterFromTypes(string[] types, bool allow)
		{
			return (IEnumerable<Token> tokens) =>
			{
				IEnumerable<Token> filtered = new List<Token>();
				foreach (Token token in tokens)
				{
					if (types.Contains(token.Type))
					{
						if (allow)
						{
							filtered.Append(token);
						}
					}
					else
					{
						if (!allow)
						{
							filtered.Append(token);
						}
					}
				}
				return filtered;
			};
		}
		public static PostProcessorDelegate CreatePostProcessorFilterForWhitespaces()
		{
			return CreatePostProcessorFilterFromTypes(["space", "newline", "tab"], false);
		}
		#endregion

		#region Split Input Token Rules
		public static SplitInputTokenRuleDelegate CreateSplitInputRuleFromList(List<string> matches, string type)
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
		public static SplitInputTokenRuleDelegate CreateSplitInputRuleFromPredicate(Func<string, bool> predicate, string type)
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
		public static SplitInputTokenRuleDelegate CreateSplitInputRuleFromRegex(string pattern, string type)
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
		public static SplitInputTokenRuleDelegate CreateMatchAllSplitInputRule(string type)
		{
			return (IEnumerable<PartialToken> parts, PartialToken part, int index, out Token? token) =>
			{
				token = new Token(part.StartIndex, part.Value, type);
				return true;
			};
		}
		#endregion

		#region Input Splitters
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
		#endregion
	}
}
