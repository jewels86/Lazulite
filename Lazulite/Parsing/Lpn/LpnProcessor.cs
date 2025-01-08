using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;
using Lazulite.Parsing;
using static Lazulite.Parsing.AstNodes;
using static Lazulite.Parsing.GrammarRules;
using static Lazulite.Parsing.Lpn.Nodes;

namespace Lazulite.Parsing.Lpn
{
	public class LpnProcessor<T>
	{
		public List<Token> Tokenize(string input)
		{
			StandardTokenizer tokenizer = new();
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"[a-zA-Z_][a-zA-Z0-9_-]*", "identifier"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(TokenizationFunctions.StandardIntegerLiteralRegex, "integer"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(TokenizationFunctions.StandardStringLiteralRegex, "string"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(TokenizationFunctions.StandardLowercaseBooleanLiteralRegex, "bool"));
			tokenizer.AddRules(TokenizationFunctions.CreateBraceRules());
			tokenizer.AddRules(TokenizationFunctions.CreateParenthesisRules());
			tokenizer.AddRules(TokenizationFunctions.CreateBracketRules());
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"=>", "function-operator"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"\.", "member-operator"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"=", "declaration-operator"));
			tokenizer.AddRule(TokenizationFunctions.CreateRuleFromRegex(@"@", "metadata-operator"));
			tokenizer.AddRules(TokenizationFunctions.CreateWhitespaceRules());
			tokenizer.AddPostProcessor(TokenizationFunctions.CreatePostProcessorFilterForWhitespaces());

			return tokenizer.Tokenize(input).ToList();
		}
		public ProgramAstNode Parse(List<Token> tokens)
		{
			ProgramAstNode program = new([]);
			RecursiveDescentParser<Token> parser = new([], null);

			var metadataOperand = new ChoiceRule<Token>([
				new TokenRule("identifier", t => new IdentifierAstNode(t.Value)),
				new TokenRule("string", t => new LiteralAstNode(t.Value, "string")),
				new TokenRule("integer", t => new LiteralAstNode(t.Value, "integer")),
				new TokenRule("bool", t => new LiteralAstNode(t.Value, "bool")),
			]);
			var metadataRule = new SequenceRule<Token>([
				new TokenValueRule("@", t => null),
				new TokenRule("identifier", t => new IdentifierAstNode(t.Value)),
				new OptionalRule<Token>(new SequenceRule<Token>([
					metadataOperand,
					new RepetitionRule<Token>(new SequenceRule<Token>([
						new TokenRule(",", t => null),
						metadataOperand
					], nodes => nodes[1]))
				], nodes =>
				{
					if (nodes[0] is null) return new RepetitionAstNode([]);
					else if (nodes[1] is null) return new RepetitionAstNode([nodes[0]]);
					else return new RepetitionAstNode(((RepetitionAstNode)nodes[1]).Children.Append(nodes[0]).ToList());
				}))
			], nodes => new MetadataAstNode(((IdentifierAstNode)nodes[1]).Name, ((RepetitionAstNode)nodes[2]).Children));

			var declarationRule = new SequenceRule<Token>([
				new TokenRule("identifier", t => new IdentifierAstNode(t.Value)),
				new TokenValueRule("=", t => null),
			])

			var programRule = new RepetitionRule<Token>(new ChoiceRule<Token>([
				metadataRule
			]));

			parser.AddRules([programRule, metadataRule]);

			var _program = parser.Parse(new(tokens));
			if (_program is ProgramAstNode) program = (ProgramAstNode)_program;
			program.Traverse(Console.WriteLine);

			return program;
		}

		public List<IGrammarRule<T>> Process(string[] paths, LpnContext<T> context)
		{
			List<LpnFile> files = [];

			foreach (string path in paths)
			{
				string content = File.ReadAllText(path);
				List<Token> tokens = Tokenize(content);
				IAstNode ast = Parse(tokens);
				files.Add(new LpnFile(path, tokens, ast));
			}

			return [];
		}
	}
}