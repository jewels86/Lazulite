using MRL.Compiler.Commands;
using Spectre.Console.Cli;

namespace Lazulite.Compiler;

public static class Program
{
    public static void Main(string[] args)
    {
        var app = new CommandApp();
        app.Configure(config =>
        {
            config.AddCommand<DefaultCommand>("compile")
                .WithDescription("Compile a MRL source file.")
                .WithExample("compile", "path/to/source.mrl");
        });
        app.Run(args);
    }
}