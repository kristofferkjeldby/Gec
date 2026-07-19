using Gec.Core.Models;
using Gec.Tests;

namespace Gec.Tests.Models;

public class LinearTests
{
    [Test]
    public void Forward_MatchesManualMatmulPlusBias()
    {
        var linear = new Linear(2, 3);
        ReflectionTestHelpers.SetLinearWeights(
            linear,
            new double[,] { { 1, 2, 3 }, { 4, 5, 6 } },
            new double[] { 10, 20, 30 });

        var input = new double[,] { { 1, 0 }, { 0, 1 } };

        var result = linear.Forward(input);

        var expected = new double[,] { { 11, 22, 33 }, { 14, 25, 36 } };
        for (var i = 0; i < 2; i++)
        for (var j = 0; j < 3; j++)
            Assert.That(result[i, j], Is.EqualTo(expected[i, j]).Within(1e-9), $"Mismatch at [{i},{j}]");
    }

    [Test]
    public void Forward_BroadcastsBiasAcrossAllRows()
    {
        var linear = new Linear(1, 2);
        ReflectionTestHelpers.SetLinearWeights(
            linear,
            new double[,] { { 0, 0 } },
            new double[] { 5, -5 });

        var input = new double[,] { { 1 }, { 2 }, { 3 } };

        var result = linear.Forward(input);

        for (var i = 0; i < 3; i++)
        {
            Assert.That(result[i, 0], Is.EqualTo(5).Within(1e-9));
            Assert.That(result[i, 1], Is.EqualTo(-5).Within(1e-9));
        }
    }

    [Test]
    public void Forward_OutputShapeMatchesSeqLenAndOutputDim()
    {
        var linear = new Linear(4, 6);

        var result = linear.Forward(new double[3, 4]);

        Assert.That(result.GetLength(0), Is.EqualTo(3));
        Assert.That(result.GetLength(1), Is.EqualTo(6));
    }

    [Test]
    public void Backward_MatchesHandComputedGradients()
    {
        var linear = new Linear(2, 3);
        ReflectionTestHelpers.SetLinearWeights(
            linear,
            new double[,] { { 1, 2, 3 }, { 4, 5, 6 } },
            new double[] { 0, 0, 0 });

        var input = new double[,] { { 1, 0 }, { 0, 1 } };
        var gradOutput = new double[,] { { 1, 0, 1 }, { 0, 1, 0 } };

        var (gradInput, gradWeights, gradBias) = linear.Backward(input, gradOutput);

        // gradInput = gradOutput * weights^T
        var expectedGradInput = new double[,] { { 4, 10 }, { 2, 5 } };
        // gradWeights = input^T * gradOutput (input is the identity here, so gradWeights == gradOutput)
        var expectedGradWeights = new double[,] { { 1, 0, 1 }, { 0, 1, 0 } };
        // gradBias = column sums of gradOutput
        var expectedGradBias = new double[] { 1, 1, 1 };

        for (var i = 0; i < 2; i++)
        for (var j = 0; j < 2; j++)
            Assert.That(gradInput[i, j], Is.EqualTo(expectedGradInput[i, j]).Within(1e-9), $"gradInput mismatch at [{i},{j}]");

        for (var i = 0; i < 2; i++)
        for (var j = 0; j < 3; j++)
            Assert.That(gradWeights[i, j], Is.EqualTo(expectedGradWeights[i, j]).Within(1e-9), $"gradWeights mismatch at [{i},{j}]");

        for (var j = 0; j < 3; j++)
            Assert.That(gradBias[j], Is.EqualTo(expectedGradBias[j]).Within(1e-9), $"gradBias mismatch at [{j}]");
    }

    [Test]
    public void Backward_GradientShapesMatchInputAndWeightShapes()
    {
        var linear = new Linear(4, 6);

        var (gradInput, gradWeights, gradBias) = linear.Backward(new double[3, 4], new double[3, 6]);

        Assert.That(gradInput.GetLength(0), Is.EqualTo(3));
        Assert.That(gradInput.GetLength(1), Is.EqualTo(4));
        Assert.That(gradWeights.GetLength(0), Is.EqualTo(4));
        Assert.That(gradWeights.GetLength(1), Is.EqualTo(6));
        Assert.That(gradBias.Length, Is.EqualTo(6));
    }
}
