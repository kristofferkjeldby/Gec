using Gec.Tokenizer.Extensions;

namespace Gec.Tokenizer.Helpers;

internal static class BpeTrainer
{
    public static (Dictionary<string, int> Vocabulary, List<string> Merges) Train(string corpus, int numMerges)
    {
        var text = corpus.PreTokenize();

        // Base vocabulary: every distinct character, in a deterministic (code-point) order.
        var vocabulary = new Dictionary<string, int>();
        foreach (var character in text.Distinct().OrderBy(c => c))
            vocabulary[character.ToString()] = vocabulary.Count;

        var symbols = text.Select(c => c.ToString()).ToList();
        var merges = new List<string>();

        for (var m = 0; m < numMerges; m++)
        {
            var best = MostFrequentPair(symbols);
            if (best is null)
                break; // no adjacent pair occurs more than once — nothing worth merging

            var (first, second) = best.Value;
            var merged = first + second;

            if (!vocabulary.ContainsKey(merged))
                vocabulary[merged] = vocabulary.Count;
            merges.Add($"{first}{Constants.SpaceString}{second}");

            symbols = ApplyMerge(symbols, first, second, merged);
        }

        return (vocabulary, merges);
    }

    public static (Dictionary<string, int> Vocabulary, List<string> Merges) TrainAndSave(string corpus, int numMerges, string outputDirectory)
    {
        var (vocabulary, merges) = Train(corpus, numMerges);

        Directory.CreateDirectory(outputDirectory);
        FileHelper.WriteVocab(vocabulary, Path.Combine(outputDirectory, Constants.VocabularyFileName));
        FileHelper.WriteMergeRules(merges, Path.Combine(outputDirectory, Constants.MergesFileName));

        return (vocabulary, merges);
    }

    // Returns the adjacent pair with the highest frequency, breaking ties by earliest occurrence so the
    // result is fully deterministic. Returns null when no pair repeats.
    private static (string First, string Second)? MostFrequentPair(List<string> symbols)
    {
        var counts = new Dictionary<(string, string), int>();
        var firstIndex = new Dictionary<(string, string), int>();

        for (var i = 0; i < symbols.Count - 1; i++)
        {
            var pair = (symbols[i], symbols[i + 1]);
            counts[pair] = counts.GetValueOrDefault(pair) + 1;
            firstIndex.TryAdd(pair, i);
        }

        (string, string)? best = null;
        foreach (var pair in counts.Keys)
        {
            if (best is null
                || counts[pair] > counts[best.Value]
                || (counts[pair] == counts[best.Value] && firstIndex[pair] < firstIndex[best.Value]))
                best = pair;
        }

        return best is not null && counts[best.Value] >= 2 ? best : null;
    }

    // Fuses every non-overlapping (first, second) occurrence into merged, left to right — matching
    // the single-pass behaviour of MergeRule.Apply used at encode time.
    private static List<string> ApplyMerge(List<string> symbols, string first, string second, string merged)
    {
        var result = new List<string>(symbols.Count);

        for (var i = 0; i < symbols.Count; i++)
        {
            if (i < symbols.Count - 1 && symbols[i] == first && symbols[i + 1] == second)
            {
                result.Add(merged);
                i++;
            }
            else
            {
                result.Add(symbols[i]);
            }
        }

        return result;
    }
}
