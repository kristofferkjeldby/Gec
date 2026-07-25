using System.Diagnostics;
using Gec.Console.Configuration;
using Gec.Console.Helpers;
using Gec.Core.Models;
using Gec.Core.Serialization;
using Gec.Core.Training;
using Gec.Tokenizer;
using Gec.Tokenizer.Helpers;
using SystemConsole = System.Console;

namespace Gec.Console.Commands;

internal static class TrainCommand
{
    public static int Run(string[] args)
    {
        var tokensPath = PathHelper.Resolve(args, 0, PathHelper.DefaultTokens);
        var vocabsDirectory = PathHelper.Resolve(args, 1, PathHelper.DefaultVocabsDirectory);
        var modelPath = PathHelper.Resolve(args, 2, PathHelper.DefaultModel);

        var steps = Settings.Training.DefaultSteps;
        if (args.Length > 3 && (!int.TryParse(args[3], out steps) || steps < 1))
        {
            SystemConsole.Error.WriteLine($"The step count must be a positive integer, but was '{args[3]}'.");
            return 1;
        }

        var vocabularyPath = Path.Combine(vocabsDirectory, Constants.VocabularyFileName);

        foreach (var required in new[] { tokensPath, vocabularyPath })
        {
            if (File.Exists(required))
                continue;

            SystemConsole.Error.WriteLine($"File not found: {required}");
            return 1;
        }

        var vocabularySize = FileHelper.ReadVocab(vocabularyPath).Count;
        var tokens = TokenFile.Read(tokensPath);

        var outOfRange = tokens.FirstOrDefault(token => token < 0 || token >= vocabularySize, -1);
        if (outOfRange >= 0)
        {
            SystemConsole.Error.WriteLine($"{tokensPath} contains token {outOfRange}, which is outside the vocabulary of {vocabularySize} tokens.");
            SystemConsole.Error.WriteLine("The tokens and the vocabulary come from different runs — re-run 'tokenize'.");
            return 1;
        }

        var minimumTokens = (int)Math.Ceiling((Settings.Model.MaxSeqLen + 1) / Settings.Training.ValidationFraction);
        if (tokens.Length < minimumTokens)
        {
            SystemConsole.Error.WriteLine($"{tokensPath} holds {tokens.Length} tokens, but at least {minimumTokens} are needed for a {Settings.Model.MaxSeqLen}-token window plus a validation split.");
            return 1;
        }

        var splitIndex = (int)(tokens.Length * (1 - Settings.Training.ValidationFraction));
        var trainTokens = tokens[..splitIndex];
        var validationTokens = tokens[splitIndex..];

        var trainPath = PathHelper.FromRepositoryRoot(PathHelper.DefaultTrainTokens);
        var validationPath = PathHelper.FromRepositoryRoot(PathHelper.DefaultValidationTokens);
        TokenFile.Write(trainPath, trainTokens);
        TokenFile.Write(validationPath, validationTokens);

        var config = new GptConfig(
            VocabSize: vocabularySize,
            DModel: Settings.Model.DModel,
            NHeads: Settings.Model.NHeads,
            NLayers: Settings.Model.NLayers,
            DFf: Settings.Model.DFf,
            MaxSeqLen: Settings.Model.MaxSeqLen);

        var random = new Random(Settings.Training.Seed);
        var trainer = new ParallelTrainer(config, Settings.Training.BatchSize, random, Settings.Training.Workers);
        var model = trainer.Model;
        var optimizer = new Adam(model.Parameters(), Settings.Training.LearningRate);

        SystemConsole.WriteLine($"Tokens     {tokensPath} ({tokens.Length} tokens, {trainTokens.Length} train / {validationTokens.Length} validation)");
        SystemConsole.WriteLine($"Written    {trainPath}");
        SystemConsole.WriteLine($"Written    {validationPath}");
        SystemConsole.WriteLine($"Model      {config.NLayers} layers, dModel {config.DModel}, {config.NHeads} heads, vocab {config.VocabSize} ({model.ParameterCount():N0} parameters)");
        SystemConsole.WriteLine($"Training   {steps} steps, batch {Settings.Training.BatchSize}, learning rate {Settings.Training.LearningRate}");
        SystemConsole.WriteLine();

        var stopwatch = Stopwatch.StartNew();

        for (var step = 1; step <= steps; step++)
        {
            var batch = NextBatch(trainTokens, random, config.MaxSeqLen, Settings.Training.BatchSize);

            var totalLoss = trainer.Backpropagate(batch);

            optimizer.ClipGradients(Settings.Training.MaxGradientNorm * Settings.Training.BatchSize);
            optimizer.Step();
            trainer.BroadcastWeights();

            if (step % Settings.Training.ReportEvery != 0 && step != steps)
                continue;

            var trainLoss = totalLoss / Settings.Training.BatchSize;
            var validationLoss = Evaluate(trainer, validationTokens, config.MaxSeqLen);

            SystemConsole.WriteLine(
                $"  step {step,5}/{steps}   train loss {trainLoss:0.0000}   validation loss {validationLoss:0.0000}   {stopwatch.Elapsed.TotalSeconds:0.0}s");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        ModelSerializer.Save(model, modelPath);

        SystemConsole.WriteLine();
        SystemConsole.WriteLine($"Written    {modelPath} ({new FileInfo(modelPath).Length / 1024:N0} KB)");

        return 0;
    }

    private static (int[] Inputs, int[] Targets) NextWindow(int[] tokens, Random random, int seqLen)
    {
        var length = Math.Min(seqLen, tokens.Length - 1);
        var start = random.Next(tokens.Length - length);

        return (tokens[start..(start + length)], tokens[(start + 1)..(start + length + 1)]);
    }

    private static List<(int[] Inputs, int[] Targets)> NextBatch(int[] tokens, Random random, int seqLen, int count)
    {
        return Enumerable.Range(0, count).Select(_ => NextWindow(tokens, random, seqLen)).ToList();
    }

    private static double Evaluate(ParallelTrainer trainer, int[] tokens, int seqLen)
    {
        var batch = NextBatch(tokens, new Random(Settings.Training.Seed), seqLen, Settings.Training.ValidationBatches);

        return trainer.Evaluate(batch) / batch.Count;
    }
}
