using Gec.Core.Models;
using Gec.Core.Training;

namespace Gec.Tests.Core.Models;

public class GptModelTests
{
    private static GptConfig TinyConfig => new(VocabSize: 7, DModel: 4, NHeads: 2, NLayers: 2, DFf: 6, MaxSeqLen: 5);

    [Test]
    public void Forward_ReturnsOneLogitRowPerTokenAndOneColumnPerVocabularyEntry()
    {
        var model = new GptModel(TinyConfig, new Random(1));

        var logits = model.Forward([1, 2, 3]);

        Assert.That(logits.GetLength(0), Is.EqualTo(3));
        Assert.That(logits.GetLength(1), Is.EqualTo(7));
    }

    [Test]
    public void Forward_ThrowsWhenSequenceExceedsMaxSeqLen()
    {
        var model = new GptModel(TinyConfig, new Random(1));

        Assert.Throws<ArgumentException>(() => model.Forward([0, 1, 2, 3, 4, 5]));
    }

    [Test]
    public void Forward_IsCausal_AppendingTokensLeavesEarlierPredictionsUnchanged()
    {
        var model = new GptModel(TinyConfig, new Random(7));

        var shortLogits = model.Forward([3, 1]);
        var longLogits = model.Forward([3, 1, 5, 2]);

        for (var row = 0; row < 2; row++)
        for (var col = 0; col < 7; col++)
            Assert.That(longLogits[row, col], Is.EqualTo(shortLogits[row, col]).Within(1e-12), $"Row {row} changed when later tokens were appended");
    }

    [Test]
    public void Backpropagate_MatchesNumericalGradientForEveryParameter()
    {
        var model = new GptModel(TinyConfig, new Random(42));
        int[] tokens = [1, 4, 2, 6];
        int[] targets = [4, 2, 6, 3];

        double Loss()
        {
            var (loss, _) = CrossEntropy.Forward(model.Forward(tokens), targets);
            return loss;
        }

        var parameters = model.Parameters().ToArray();
        foreach (var parameter in parameters)
            parameter.ZeroGradient();

        var (_, gradLogits) = CrossEntropy.Forward(model.Forward(tokens), targets);
        model.Backpropagate(gradLogits);

        foreach (var parameter in parameters)
            NumericalGradient.AssertParameterGradient(parameter, Loss);
    }

    [Test]
    public void Backpropagate_AccumulatesAcrossSequences()
    {
        var model = new GptModel(TinyConfig, new Random(3));
        int[] tokens = [1, 4, 2];
        int[] targets = [4, 2, 6];

        var parameter = model.Parameters().First();

        parameter.ZeroGradient();
        var (_, gradLogits) = CrossEntropy.Forward(model.Forward(tokens), targets);
        model.Backpropagate(gradLogits);
        var single = Enumerable.Range(0, parameter.Count).Select(parameter.GetGradient).ToArray();

        var (_, gradLogitsAgain) = CrossEntropy.Forward(model.Forward(tokens), targets);
        model.Backpropagate(gradLogitsAgain);
        var doubled = Enumerable.Range(0, parameter.Count).Select(parameter.GetGradient).ToArray();

        for (var i = 0; i < single.Length; i++)
            Assert.That(doubled[i], Is.EqualTo(2 * single[i]).Within(1e-12));
    }

    [Test]
    public void ParameterNames_AreUnique()
    {
        var model = new GptModel(TinyConfig, new Random(1));

        var names = model.Parameters().Select(parameter => parameter.Name).ToArray();

        Assert.That(names, Is.Unique);
    }
}
