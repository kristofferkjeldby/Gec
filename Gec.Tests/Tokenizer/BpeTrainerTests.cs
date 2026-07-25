using Gec.Tokenizer.Extensions;
using Gec.Tokenizer.Helpers;
using Gec.Tokenizer.Models;

namespace Gec.Tests.Tokenizer;

public class BpeTrainerTests
{
    private const string Corpus =
        "the cat sat on the mat. the cat ran to the mat. " +
        "a cat and a rat sat on the mat. the rat ran. ";

    private static List<int> Encode(string text, IDictionary<string, int> vocabulary, List<string> merges)
    {
        var rules = merges.Select(rule => new MergeRule(rule, vocabulary)).ToList();

        return BpeEncoder.Encode(text, vocabulary, rules);
    }

    private static string Decode(List<int> ids, IDictionary<string, int> vocabulary)
    {
        return BpeEncoder.Decode(ids, vocabulary);
    }

    [Test]
    public void Train_ThenEncodeDecode_RoundTripsAndMergesShortenTheSequence()
    {
        var (vocabulary, merges) = BpeTrainer.Train(Corpus, numMerges: 30);

        // Training stops early once no adjacent pair repeats, so we get at most the requested count.
        Assert.That(merges, Has.Count.GreaterThan(0).And.Count.LessThanOrEqualTo(30));

        const string sample = "the cat sat on the mat.";
        var baseLength = sample.ToInts(vocabulary).Count;
        var encoded = Encode(sample, vocabulary, merges);

        Assert.That(encoded, Has.Count.LessThan(baseLength), "merges should compress the token sequence");
        Assert.That(Decode(encoded, vocabulary), Is.EqualTo(sample), "encode then decode is lossless");
    }

    [Test]
    public void TrainAndSave_ThenReadBack_ReproducesTheSameEncoding()
    {
        var (expectedVocab, expectedMerges) = BpeTrainer.Train(Corpus, numMerges: 30);
        var expected = Encode("a rat ran to the mat.", expectedVocab, expectedMerges);

        var directory = Path.Combine(Path.GetTempPath(), "gec-bpe-test");
        try
        {
            BpeTrainer.TrainAndSave(Corpus, numMerges: 30, directory);

            var vocabulary = FileHelper.ReadVocab(Path.Combine(directory, "vocab.json"));
            var merges = FileHelper.ReadMergeRules(Path.Combine(directory, "merges.txt"), vocabulary)
                .Select(rule => rule.ToString())
                .ToList();

            var actual = Encode("a rat ran to the mat.", vocabulary, merges);

            Assert.That(actual, Is.EqualTo(expected), "files round-trip to an identical encoding");
        }
        finally
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }
}
