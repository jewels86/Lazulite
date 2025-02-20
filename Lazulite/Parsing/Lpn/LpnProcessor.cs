﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lazulite.Tokenization;
using Lazulite.Parsing;
using Lazulite.IR;
using static Lazulite.Parsing.AstNodes;
using static Lazulite.Parsing.GrammarRules;
using static Lazulite.Parsing.Lpn.Nodes;

namespace Lazulite.Parsing.Lpn
{
	public class LpnProcessor<T>
	{
		public LpnContext<T> Context { get; } = new();

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
		public void Process(string input)
		{
			List<Token> tokens = Tokenize(input);
			ProgramAstNode program = Parse(tokens);
		}
		public List<IGrammarRule<T>> ProcessFile(string path)
		{
			string content = File.ReadAllText(path);
			List<Token> tokens = Tokenize(content);
			ProgramAstNode program = Parse(tokens);

			program.Traverse(node =>
			{
				if (node is MetadataAstNode metadata) Context.AddMetadata(metadata.Key, [.. HandleLiterals(metadata.Operands)]);
				else if (node is DeclarationAstNode dec)
				{
					IGrammarRule<T> rule = null!;

					if (rule is 
				}
			});
		}

		private ProgramAstNode Parse(List<Token> tokens)
		{
			ProgramAstNode program = new([]);
			RecursiveDescentParser<Token> parser = new([], null);
			ParsingRuleset ruleset = new([]);

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

			var sequenceRule = new SequenceRule<Token>([
				new TokenValueRule("[", t => null),
				new RepetitionRule<Token>(null!), // to be injected with rule
				new TokenValueRule("]", t => null),
			], nodes => nodes[1]);
			var optionalRule = new SequenceRule<Token>([
				new TokenValueRule("(", t => null),
				null, // to be injected with rule
				new TokenValueRule(")", t => null),
			], nodes => nodes[1]);
			var repetitionRule = new SequenceRule<Token>([
				new TokenValueRule("{", t => null),
				null, // to be injected with rule
				new TokenValueRule("}", t => null),
			], nodes => nodes[1]);
			
			var stringLiteralRule = new SequenceRule<Token>([
			    new TokenValueRule(
			]);

			var rule = new ChoiceRule<Token>([sequenceRule, optionalRule, repetitionRule, ]);

			sequenceRule[1][0] = rule;
			optionalRule[1] = rule;
			repetitionRule[1] = rule;

			var declarationRule = new SequenceRule<Token>([
				new TokenRule("identifier", t => new IdentifierAstNode(t.Value)),
				new TokenValueRule("=", t => null),
				rule,
				new OptionalRule<Token>(new SequenceRule<Token>([
					new TokenValueRule("=>", t => null),
					null // block rule
				], nodes => nodes[1]))
			], nodes =>
			{
				if (nodes[3] is not null) return new DeclarationAstNode(nodes[0], nodes[2], nodes[3]);
				else return new DeclarationAstNode(nodes[0], nodes[2], null);
			});

			var scriptAssignmentRule = new SequenceRule<Token>([
				new TokenRule("identifier", t => new IdentifierAstNode(t.Value)),
				new TokenValueRule("=", t => null),
				ruleset.ExpressionRule
			], nodes => new StaticAssignmentAstNode(nodes[0], nodes[2]));
			var scriptDefinitionRule = new SequenceRule<Token>([
				new TokenRule("type", t => new TypeAstNode(t.Value)),
				scriptAssignmentRule
			], nodes => new ScriptDefinitionAstNode(nodes[0], nodes[0][0], nodes[0][1]));

			var statementRule = new ChoiceRule<Token>([metadataRule, declarationRule]);
			var programRule = new RepetitionRule<Token>(statementRule);

			parser.AddRules([programRule, metadataRule]);

			var _program = parser.Parse(new(tokens));
			if (_program is ProgramAstNode) program = (ProgramAstNode)_program;
			program.Traverse(Console.WriteLine);

			return program;
		}
		private List<string> HandleLiterals(List<IAstNode> nodes)
		{
			return nodes.Select(node =>
			{
				if (node is not LiteralAstNode) throw new ArgumentException($"Expected literal, got {node}");
				return ((LiteralAstNode)node).Value;
			}).ToList();
		}
	}
}