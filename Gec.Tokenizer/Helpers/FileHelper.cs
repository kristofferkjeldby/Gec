using Gec.Tokenizer.Extensions;
using Gec.Tokenizer.Models;

namespace Gec.Tokenizer.Helpers;

internal static class FileHelper
{
    public static IEnumerable<MergeRule> ReadMergeRules(string filePath, IDictionary<string, int> vocabulary)
    {
        return ReadMergeRules(File.ReadLines(filePath).ToArray(), vocabulary);
    }

    private static IEnumerable<MergeRule> ReadMergeRules(string[] lines, IDictionary<string, int> vocabulary)
    {
        var mergeRules = new List<MergeRule>();

        for (var i = 0; i < lines.Count(); i++)
        {
            if (lines[i].StartsWith(Constants.Comment))
                continue;
            mergeRules.Add(new MergeRule(lines[i], vocabulary));
        }

        return mergeRules;
    }

    // Writes merge rules in the same "#version" + space-delimited "A B" format ReadMergeRules parses.
    public static void WriteMergeRules(IEnumerable<string> merges, string filePath)
    {
        var lines = new List<string> { $"{Constants.Comment}version: gec-bpe-1" };
        lines.AddRange(merges);
        File.WriteAllLines(filePath, lines);
    }

    // Writes the vocabulary as the JSON object ReadVocabulary parses (one "token": id entry per line).
    public static void WriteVocab(IDictionary<string, int> vocabulary, string filePath)
    {
        var ordered = vocabulary.OrderBy(entry => entry.Value).ToList();

        var lines = new List<string> { "{" };
        for (var i = 0; i < ordered.Count; i++)
        {
            var token = ordered[i].Key.Replace("\\", "\\\\").Replace("\"", "\\\"");
            var comma = i < ordered.Count - 1 ? Constants.Comma.ToString() : string.Empty;
            lines.Add($"  {Constants.Quote}{token}{Constants.Quote}{Constants.ColonString} {ordered[i].Value}{comma}");
        }
        lines.Add("}");

        File.WriteAllLines(filePath, lines);
    }

    public static Dictionary<string, int> ReadVocab(string filePath)
    {
        return ReadVocab(File.ReadLines(filePath).ToArray());
    }

    private static Dictionary<string, int> ReadVocab(string[] lines)
    {
        var vocabulary = new Dictionary<string, int>();

        for (var i = 0; i < lines.Count(); i++)
        {
            var parts = lines[i].Trim().Split(Constants.Colon).ToArray();

            if (parts.Length < 2)
                continue;

            if (!int.TryParse(parts.Last().Trim(Constants.Space, Constants.Comma), out var id))
                continue;

            var value = string.Join(Constants.ColonString, parts.Take(parts.Length - 1)).Trim(Constants.Space);

            vocabulary.Add(value.Substring(1, value.Length - 2).JsonUnescape(), id);
        }

        return vocabulary;
    }
}
