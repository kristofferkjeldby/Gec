using Gec.Console.Helpers;
using Gec.Tokenizer;
using Gec.Tokenizer.Helpers;
using SystemConsole = System.Console;

namespace Gec.Console.Commands;

internal static class TokenizeCommand
{
    private const int TokensPerLine = 20;

    public static int Run(string[] args)
    {
        var corpusPath = PathHelper.Resolve(args, 0, PathHelper.DefaultCorpus);
        var vocabsDirectory = PathHelper.Resolve(args, 1, PathHelper.DefaultVocabsDirectory);
        var outputPath = PathHelper.Resolve(args, 2, PathHelper.DefaultTokens);

        var vocabularyPath = Path.Combine(vocabsDirectory, Constants.VocabularyFileName);
        var mergesPath = Path.Combine(vocabsDirectory, Constants.MergesFileName);

        foreach (var required in new[] { corpusPath, vocabularyPath, mergesPath })
        {
            if (File.Exists(required))
                continue;

            SystemConsole.Error.WriteLine($"File not found: {required}");
            return 1;
        }

        var corpus = File.ReadAllText(corpusPath);
        if (corpus.Length == 0)
        {
            SystemConsole.Error.WriteLine($"Corpus is empty: {corpusPath}");
            return 1;
        }

        var vocabulary = FileHelper.ReadVocab(vocabularyPath);
        var mergeRules = FileHelper.ReadMergeRules(mergesPath, vocabulary).ToList();

        List<int> tokens;
        try
        {
            tokens = BpeEncoder.Encode(corpus, vocabulary, mergeRules);
        }
        catch (KeyNotFoundException)
        {
            // ToInts looks every character up directly — there is no unknown-token fallback yet.
            SystemConsole.Error.WriteLine("The corpus contains characters that are missing from the vocabulary.");
            SystemConsole.Error.WriteLine($"Run 'vocab' on this corpus first to retrain {vocabularyPath}.");
            return 1;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllLines(outputPath, tokens.Chunk(TokensPerLine).Select(chunk => string.Join(' ', chunk)));

        SystemConsole.WriteLine($"Corpus     {corpusPath} ({corpus.Length} characters)");
        SystemConsole.WriteLine($"Vocabulary {vocabularyPath} ({vocabulary.Count} tokens, {mergeRules.Count} merges)");
        SystemConsole.WriteLine($"Encoded    {tokens.Count} tokens ({(double)corpus.Length / tokens.Count:0.00} characters per token)");
        SystemConsole.WriteLine($"Written    {outputPath}");

        return 0;
    }
}
