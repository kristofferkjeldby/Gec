using Gec.Console.Configuration;
using Gec.Console.Helpers;
using Gec.Core.Common;
using Gec.Core.Serialization;
using Gec.Tokenizer;
using Gec.Tokenizer.Helpers;
using SystemConsole = System.Console;

namespace Gec.Console.Commands;

internal static class CompleteCommand
{
    public static int Run(string[] args)
    {
        if (args.Length == 0 || args[0].Length == 0)
        {
            SystemConsole.Error.WriteLine("A prompt is required: complete <prompt> [max-tokens] [model] [vocab-directory]");
            return 1;
        }

        var prompt = args[0];
        var modelPath = PathHelper.Resolve(args, 2, PathHelper.DefaultModel);
        var vocabsDirectory = PathHelper.Resolve(args, 3, PathHelper.DefaultVocabsDirectory);

        var maxTokens = Settings.Completion.DefaultMaxTokens;
        if (args.Length > 1 && (!int.TryParse(args[1], out maxTokens) || maxTokens < 1))
        {
            SystemConsole.Error.WriteLine($"The token count must be a positive integer, but was '{args[1]}'.");
            return 1;
        }

        var vocabularyPath = Path.Combine(vocabsDirectory, Constants.VocabularyFileName);
        var mergesPath = Path.Combine(vocabsDirectory, Constants.MergesFileName);

        foreach (var required in new[] { modelPath, vocabularyPath, mergesPath })
        {
            if (File.Exists(required))
                continue;

            SystemConsole.Error.WriteLine($"File not found: {required}");
            if (required == modelPath)
                SystemConsole.Error.WriteLine("Run 'train' first to produce a model.");
            return 1;
        }

        var model = ModelSerializer.Load(modelPath);
        var vocabulary = FileHelper.ReadVocab(vocabularyPath);
        var mergeRules = FileHelper.ReadMergeRules(mergesPath, vocabulary).ToList();

        if (vocabulary.Count != model.Config.VocabSize)
        {
            SystemConsole.Error.WriteLine($"The model was trained on a vocabulary of {model.Config.VocabSize} tokens, but {vocabularyPath} holds {vocabulary.Count}.");
            return 1;
        }

        List<int> tokens;
        try
        {
            tokens = BpeEncoder.Encode(prompt, vocabulary, mergeRules);
        }
        catch (KeyNotFoundException)
        {
            SystemConsole.Error.WriteLine("The prompt contains characters that are missing from the vocabulary.");
            return 1;
        }

        if (tokens.Count == 0)
        {
            SystemConsole.Error.WriteLine("The prompt encoded to no tokens.");
            return 1;
        }

        var random = new Random(Settings.Completion.Seed);
        var generated = new List<int>();

        for (var i = 0; i < maxTokens; i++)
        {
            var context = tokens.Count > model.Config.MaxSeqLen
                ? tokens.GetRange(tokens.Count - model.Config.MaxSeqLen, model.Config.MaxSeqLen)
                : tokens;

            var logits = model.Forward(context.ToArray());
            var next = SampleLastRow(logits, random, Settings.Completion.Temperature);

            tokens.Add(next);
            generated.Add(next);
        }

        SystemConsole.WriteLine($"Model      {modelPath} ({model.Config.NLayers} layers, dModel {model.Config.DModel}, vocab {model.Config.VocabSize})");
        SystemConsole.WriteLine($"Prompt     {prompt}");
        SystemConsole.WriteLine($"Generated  {generated.Count} tokens at temperature {Settings.Completion.Temperature}");
        SystemConsole.WriteLine();
        SystemConsole.WriteLine(BpeEncoder.Decode(tokens, vocabulary));

        return 0;
    }

    private static int SampleLastRow(double[,] logits, Random random, double temperature)
    {
        var lastRow = logits.GetLength(0) - 1;
        var vocabSize = logits.GetLength(1);

        var scaled = new double[vocabSize];
        for (var c = 0; c < vocabSize; c++)
            scaled[c] = logits[lastRow, c] / temperature;

        var probabilities = Softmax.Forward(scaled);

        var threshold = random.NextDouble();
        double cumulative = 0;

        for (var c = 0; c < vocabSize; c++)
        {
            cumulative += probabilities[c];
            if (cumulative >= threshold)
                return c;
        }

        return vocabSize - 1;
    }
}
