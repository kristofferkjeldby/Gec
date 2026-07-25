using Gec.Tokenizer.Extensions;
using Gec.Tokenizer.Models;

namespace Gec.Tokenizer.Helpers;

internal static class BpeEncoder
{
    public static List<int> Encode(string text, IDictionary<string, int> vocabulary, IEnumerable<MergeRule> mergeRules)
    {
        var tokens = text.ToInts(vocabulary);

        foreach (var mergeRule in mergeRules)
            mergeRule.Apply(tokens);

        return tokens;
    }

    public static string Decode(List<int> tokens, IDictionary<string, int> vocabulary)
    {
        var reverseVocabulary = vocabulary.ToDictionary(entry => entry.Value, entry => entry.Key);

        return tokens.FromInts(reverseVocabulary).Detokenize();
    }
}
