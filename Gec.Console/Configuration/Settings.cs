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

    public static class Model
    {
        public const int DModel = 96;

        public const int NHeads = 6;

        public const int NLayers = 4;

        public const int DFf = 384;

        public const int MaxSeqLen = 256;
    }

    public static class Training
    {
        public const int DefaultSteps = 3300;

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

        public const int Seed = 1337;
    }
}
