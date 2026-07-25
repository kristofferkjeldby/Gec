using Gec.Console.Configuration;

namespace Gec.Console.Helpers;

internal static class PathHelper
{
    private static readonly string RepositoryRoot = FindRepositoryRoot();

    public static string DefaultCorpus => Settings.Paths.DefaultCorpus;

    public static string DefaultVocabsDirectory => Settings.Paths.DefaultVocabsDirectory;

    public static string DefaultTokens => Settings.Paths.DefaultTokens;

    public static string DefaultTrainTokens => Settings.Paths.DefaultTrainTokens;

    public static string DefaultValidationTokens => Settings.Paths.DefaultValidationTokens;

    public static string DefaultModel => Settings.Paths.DefaultModel;

    public static string Resolve(string[] args, int index, string defaultRelativePath)
    {
        return index < args.Length && !string.IsNullOrWhiteSpace(args[index])
            ? Path.GetFullPath(args[index])
            : FromRepositoryRoot(defaultRelativePath);
    }

    public static string FromRepositoryRoot(string relativePath)
    {
        return Path.GetFullPath(Path.Combine(RepositoryRoot, relativePath));
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, Settings.Paths.SolutionFileName)))
            directory = directory.Parent;

        return directory?.FullName ?? Directory.GetCurrentDirectory();
    }
}
