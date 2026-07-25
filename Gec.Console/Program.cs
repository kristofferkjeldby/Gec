using Gec.Console.Commands;
using Gec.Console.Configuration;

if (args.Length == 0)
{
    PrintUsage();
    return 1;
}

switch (args[0].ToLowerInvariant())
{
    case "vocab":
        return VocabCommand.Run(args[1..]);

    case "tokenize":
        return TokenizeCommand.Run(args[1..]);

    case "train":
        return TrainCommand.Run(args[1..]);

    case "complete":
        return CompleteCommand.Run(args[1..]);

    default:
        Console.Error.WriteLine($"Unknown command '{args[0]}'.");
        PrintUsage();
        return 1;
}

void PrintUsage()
{
    Console.WriteLine("Usage: Gec.Console <command> [arguments]");
    Console.WriteLine();
    Console.WriteLine("  vocab    [corpus] [merges] [output-directory]");
    Console.WriteLine("           Train a BPE vocabulary on a corpus.");
    Console.WriteLine("  tokenize [corpus] [vocab-directory] [output]");
    Console.WriteLine("           Encode a corpus into token ids.");
    Console.WriteLine("  train    [tokens] [vocab-directory] [model] [steps]");
    Console.WriteLine("           Split the tokens into train/validation and train a model.");
    Console.WriteLine("  complete <prompt> [max-tokens] [model] [vocab-directory]");
    Console.WriteLine("           Generate text from a trained model.");
    Console.WriteLine();
    Console.WriteLine("An omitted or empty argument falls back to the repository layout:");
    Console.WriteLine($"  corpus            {Settings.Paths.DefaultCorpus}");
    Console.WriteLine($"  vocab directory   {Settings.Paths.DefaultVocabsDirectory}");
    Console.WriteLine($"  token output      {Settings.Paths.DefaultTokens}");
    Console.WriteLine($"  train / validation {Settings.Paths.DefaultTrainTokens}, {Settings.Paths.DefaultValidationTokens}");
    Console.WriteLine($"  model             {Settings.Paths.DefaultModel}");
    Console.WriteLine($"  merges            500");
    Console.WriteLine($"  steps             {Settings.Training.DefaultSteps}");
    Console.WriteLine($"  max-tokens        {Settings.Completion.DefaultMaxTokens}");
}
