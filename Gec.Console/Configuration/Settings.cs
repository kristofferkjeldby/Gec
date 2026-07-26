namespace Gec.Console.Configuration;

public static class Settings
{
    public static class Paths
    {
        public static string DataDirectory = "Gec.Console/Data";

        public static string DefaultCorpus = $"{DataDirectory}/corpus.txt";

        public static string DefaultVocabsDirectory = DataDirectory;

        public static string DefaultTokens = $"{DataDirectory}/tokens.txt";

        public static string DefaultTrainTokens = $"{DataDirectory}/train.txt";

        public static string DefaultValidationTokens = $"{DataDirectory}/val.txt";

        public static readonly string DefaultModel = $"{DataDirectory}/model.json";

        public const string SolutionFileName = "Gec.sln";
    }

    // Humble dev-phase budget for fast iteration (~283K params, ~2-3 min runs).
    // Chinchilla-optimal for the current 19.5M-token corpus is DModel 96 / NHeads 4 /
    // NLayers 7 / DFf 384 / MaxSeqLen 384 (~931K params) with DefaultSteps 5080 — switch
    // back to those once the design has settled and it's time for a real training run.
    public static class Model
    {
        public const int DModel = 64;

        public const int NHeads = 4;

        public const int NLayers = 4;

        public const int DFf = 256;

        public const int MaxSeqLen = 128;
    }

    public static class Training
    {
        public const int DefaultSteps = 1500;

        public const double LearningRate = 3e-3;

        public const int BatchSize = 10;

        public const double MaxGradientNorm = 1.0;

        public const double ValidationFraction = 0.1;

        public const int ValidationBatches = 10;

        public const int Workers = 10;

        public const int ReportEvery = 100;

        public const int Seed = 1337;
    }

    public static class Completion
    {
        public const int DefaultMaxTokens = 64;

        public const double Temperature = 0.8;
    }
}
