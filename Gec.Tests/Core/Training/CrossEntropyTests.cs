using Gec.Core.Training;

namespace Gec.Tests.Core.Training;

public class CrossEntropyTests
{
    [Test]
    public void Forward_UniformLogits_LossIsLogOfVocabularySize()
    {
        var logits = new double[,] { { 0, 0, 0, 0 }, { 5, 5, 5, 5 } };

        var (loss, _) = CrossEntropy.Forward(logits, [2, 0]);

        Assert.That(loss, Is.EqualTo(Math.Log(4)).Within(1e-12));
    }

    [Test]
    public void Forward_GradientSumsToZeroPerRow()
    {
        var logits = new double[,] { { 1.5, -0.3, 0.2 }, { -2, 0.7, 1.1 } };

        var (_, gradLogits) = CrossEntropy.Forward(logits, [0, 2]);

        for (var r = 0; r < 2; r++)
        {
            var sum = Enumerable.Range(0, 3).Sum(c => gradLogits[r, c]);
            Assert.That(sum, Is.EqualTo(0).Within(1e-12), $"Row {r}");
        }
    }

    [Test]
    public void Forward_GradientMatchesNumericalGradientOfLoss()
    {
        var logits = new double[,] { { 1.5, -0.3, 0.2 }, { -2, 0.7, 1.1 } };
        int[] targets = [0, 2];

        var (_, gradLogits) = CrossEntropy.Forward(logits, targets);

        NumericalGradient.AssertMatrixGradient(logits, gradLogits, () => CrossEntropy.Forward(logits, targets).Loss);
    }

    [Test]
    public void Forward_ConfidentCorrectPrediction_LossApproachesZero()
    {
        var logits = new double[,] { { 50, 0, 0 } };

        var (loss, _) = CrossEntropy.Forward(logits, [0]);

        Assert.That(loss, Is.EqualTo(0).Within(1e-9));
    }

    [Test]
    public void Forward_ThrowsWhenTargetCountDoesNotMatchRowCount()
    {
        Assert.Throws<ArgumentException>(() => CrossEntropy.Forward(new double[2, 3], [0]));
    }

    [Test]
    public void Forward_ThrowsWhenTargetIsOutsideTheVocabulary()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CrossEntropy.Forward(new double[1, 3], [3]));
    }
}
