namespace Gec.Console.Helpers;

internal static class TokenFile
{
    private const int TokensPerLine = 20;

    public static int[] Read(string filePath)
    {
        return File.ReadAllText(filePath)
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Select(int.Parse)
            .ToArray();
    }

    public static void Write(string filePath, IEnumerable<int> tokens)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        File.WriteAllLines(filePath, tokens.Chunk(TokensPerLine).Select(chunk => string.Join(' ', chunk)));
    }
}
