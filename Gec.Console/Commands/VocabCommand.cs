using Gec.Console.Helpers;
using Gec.Tokenizer;
using Gec.Tokenizer.Helpers;
using SystemConsole = System.Console;

namespace Gec.Console.Commands;

internal static class VocabCommand
{
    private const int DefaultNumMerges = 500;

    public static int Run(string[] args)
    {
        var corpusPath = PathHelper.Resolve(args, 0, PathHelper.DefaultCorpus);
        var outputDirectory = PathHelper.Resolve(args, 2, PathHelper.DefaultVocabsDirectory);

        var numMerges = DefaultNumMerges;
        if (args.Length > 1 && (!int.TryParse(args[1], out numMerges) || numMerges < 1))
        {
            SystemConsole.Error.WriteLine($"The merge count must be a positive integer, but was '{args[1]}'.");
            return 1;
        }

        if (!File.Exists(corpusPath))
        {
            SystemConsole.Error.WriteLine($"Corpus not found: {corpusPath}");
            return 1;
        }

        var corpus = File.ReadAllText(corpusPath);
        if (corpus.Length == 0)
        {
            SystemConsole.Error.WriteLine($"Corpus is empty: {corpusPath}");
            return 1;
        }

        SystemConsole.WriteLine($"Corpus     {corpusPath} ({corpus.Length} characters)");
        SystemConsole.WriteLine($"Training   {numMerges} merges...");

        var (vocabulary, merges) = BpeTrainer.TrainAndSave(corpus, numMerges, outputDirectory);

        SystemConsole.WriteLine($"Vocabulary {vocabulary.Count} tokens ({vocabulary.Count - merges.Count} characters + {merges.Count} merges)");
        SystemConsole.WriteLine($"Written    {Path.Combine(outputDirectory, Constants.VocabularyFileName)}");
        SystemConsole.WriteLine($"Written    {Path.Combine(outputDirectory, Constants.MergesFileName)}");

        return 0;
    }
}
