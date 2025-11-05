using Spectre.Console.Cli;
using MRL;

namespace Lazulite.Compiler.Commands;

public class CompileCommand : Command<CompileCommand.Settings>
{
    public class Settings : CommandSettings
    {
        [CommandArgument(0, "<file-path>")]
        public required string FilePath { get; init; }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        Console.WriteLine($"Compiling {settings.FilePath}...");
        string source = File.ReadAllText(settings.FilePath);
        MRLCompiler.Compile(source);
        return 0;
    }
}