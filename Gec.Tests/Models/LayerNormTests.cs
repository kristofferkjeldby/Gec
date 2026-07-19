using Gec.Core.Models;
using Gec.Tests;

namespace Gec.Tests.Models;

public class LayerNormTests
{
    [Test]
    public void Forward_ConstantRow_ReturnsAllZeros()
    {
        var layerNorm = new LayerNorm(4);

        var result = layerNorm.Forward(new double[,] { { 7, 7, 7, 7 } });

        for (var j = 0; j < 4; j++)
            Assert.That(result[0, j], Is.EqualTo(0.0).Within(1e-9));
    }

    [Test]
    public void Forward_MatchesHandComputedNormalization()
    {
        var layerNorm = new LayerNorm(4);

        // mean = 2.5, variance = 1.25, gamma = 1, beta = 0 (defaults)
        var result = layerNorm.Forward(new double[,] { { 1, 2, 3, 4 } });

        var expected = new[] { -1.34160, -0.44721, 0.44721, 1.34160 };
        for (var j = 0; j < expected.Length; j++)
            Assert.That(result[0, j], Is.EqualTo(expected[j]).Within(1e-4));
    }

    [Test]
    public void Forward_NormalizesEachRowIndependently()
    {
        var layerNorm = new LayerNorm(5);

        var result = layerNorm.Forward(new double[,] { { 3, -1, 4, 1, 5 }, { 10, 10, 10, 10, 20 } });

        for (var r = 0; r < 2; r++)
        {
            double mean = 0, variance = 0;
            for (var c = 0; c < 5; c++) mean += result[r, c];
            mean /= 5;
            for (var c = 0; c < 5; c++) variance += (result[r, c] - mean) * (result[r, c] - mean);
            variance /= 5;

            Assert.That(mean, Is.EqualTo(0.0).Within(1e-6), $"row {r} mean");
            Assert.That(variance, Is.EqualTo(1.0).Within(1e-3), $"row {r} variance");
        }
    }

    [Test]
    public void Forward_OutputShapeMatchesInputShape()
    {
        var layerNorm = new LayerNorm(6);

        var result = layerNorm.Forward(new double[3, 6]);

        Assert.That(result.GetLength(0), Is.EqualTo(3));
        Assert.That(result.GetLength(1), Is.EqualTo(6));
    }

    [Test]
    public void Backward_MatchesNumericalGradientOfForward()
    {
        var layerNorm = new LayerNorm(3);

        var input = new double[,] { { 1, 2, 3 }, { 4, 0, -2 } };
        var gradOutput = new double[,] { { 1, -1, 0.5 }, { 0.2, 0.3, -0.7 } };

        layerNorm.Forward(input);
        var (gradInput, gradGamma, gradBeta) = layerNorm.Backward(gradOutput);

        double Loss(double[,] x) => NumericalGradient.DotProduct(layerNorm.Forward(x), gradOutput);

        NumericalGradient.AssertMatrixGradient(input, gradInput, () => Loss(input));

        var gamma = ReflectionTestHelpers.GetField<double[]>(layerNorm, "_gamma");
        var beta = ReflectionTestHelpers.GetField<double[]>(layerNorm, "_beta");

        NumericalGradient.AssertVectorGradient(gamma, gradGamma, () => Loss(input));
        NumericalGradient.AssertVectorGradient(beta, gradBeta, () => Loss(input));
    }
}
