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

    // Scaled back down for fast iteration (~282K params, dModel 64 / 4 layers / 4 heads).
    // Chinchilla-optimal for the current 19.5M-token corpus is DModel 96 / NHeads 4 /
    // NLayers 7 / DFf 384 / MaxSeqLen 384 (~931K params) with DefaultSteps 5080 — switch
    // back to those for a real training run. MaxSeqLen is kept short here since it's the
    // biggest lever on wall-clock time (attention cost grows with its square).
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
        // 17,559,211 train tokens / (BatchSize 10 * MaxSeqLen 128) ≈ one full epoch.
        public const int DefaultSteps = 13718;

        public const double LearningRate = 3e-3;

        public const int BatchSize = 10;

        public const double MaxGradientNorm = 1.0;

        public const double ValidationFraction = 0.1;

        public const int ValidationBatches = 10;

        public const int Workers = 10;

        public const int ReportEvery = 10;

        public const int Seed = 1337;

        // Stop early once validation loss hasn't improved by at least MinDelta for Patience
        // consecutive reports (i.e. Patience * ReportEvery steps) — validation loss is noisy
        // step to step, so both need to span enough reports to not trip on that noise.
        public const int EarlyStoppingPatience = 50;

        public const double EarlyStoppingMinDelta = 0.005;
    }

    public static class Completion
    {
        public const int DefaultMaxTokens = 64;

        public const double Temperature = 0.8;
    }
}
