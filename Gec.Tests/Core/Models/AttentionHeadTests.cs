using Gec.Core.Models;
using Gec.Tests;

namespace Gec.Tests.Models;

public class AttentionHeadTests
{
    [Test]
    public void Forward_MatchesHandComputedScaledDotProductAttention()
    {
        // dModel = dHead = 2, Q/K/V weights are the identity with zero bias, so Q = K = V = input.
        var head = new AttentionHead(dModel: 2, dHead: 2);
        var identity = new double[,] { { 1, 0 }, { 0, 1 } };
        var zeroBias = new double[] { 0, 0 };

        ReflectionTestHelpers.SetLinearWeights(ReflectionTestHelpers.GetField<Linear>(head, "_query"), identity, zeroBias);
        ReflectionTestHelpers.SetLinearWeights(ReflectionTestHelpers.GetField<Linear>(head, "_key"), identity, zeroBias);
        ReflectionTestHelpers.SetLinearWeights(ReflectionTestHelpers.GetField<Linear>(head, "_value"), identity, zeroBias);

        var input = new double[,] { { 1, 0 }, { 0, 1 } };

        var result = head.Forward(input);

        // scores = X * X^T = I, scaled by 1/sqrt(2), softmax per row gives:
        // [exp(1/sqrt2), exp(0)] normalized -> [0.6697615, 0.3302385] and its mirror image.
        var expected = new double[,]
        {
            { 0.6697615493266569, 0.3302384506733431 },
            { 0.3302384506733431, 0.6697615493266569 }
        };

        for (var i = 0; i < 2; i++)
        for (var j = 0; j < 2; j++)
            Assert.That(result[i, j], Is.EqualTo(expected[i, j]).Within(1e-9), $"Mismatch at [{i},{j}]");
    }

    [Test]
    public void Forward_ZeroQueryAndKeyWeights_ProducesUniformAverageOfValues()
    {
        // With Q = K = 0, every score is 0, so softmax is uniform: the output is just the
        // row-wise average of V regardless of the input content.
        var head = new AttentionHead(dModel: 2, dHead: 2);
        var query = ReflectionTestHelpers.GetField<Linear>(head, "_query");
        var key = ReflectionTestHelpers.GetField<Linear>(head, "_key");
        var value = ReflectionTestHelpers.GetField<Linear>(head, "_value");

        ReflectionTestHelpers.ZeroOutLinear(query);
        ReflectionTestHelpers.ZeroOutLinear(key);
        ReflectionTestHelpers.SetLinearWeights(value, new double[,] { { 0, 0 }, { 0, 0 } }, new double[] { 3, 4 });

        var input = new double[,] { { 100, -50 }, { 2, 9 }, { -7, 3 } };

        var result = head.Forward(input);

        for (var i = 0; i < 3; i++)
        {
            Assert.That(result[i, 0], Is.EqualTo(3).Within(1e-9));
            Assert.That(result[i, 1], Is.EqualTo(4).Within(1e-9));
        }
    }

    [Test]
    public void Forward_OutputShapeIsSeqLenByDHead()
    {
        var head = new AttentionHead(dModel: 6, dHead: 4);

        var result = head.Forward(new double[5, 6]);

        Assert.That(result.GetLength(0), Is.EqualTo(5));
        Assert.That(result.GetLength(1), Is.EqualTo(4));
    }

    [Test]
    public void Backward_MatchesNumericalGradientOfForward()
    {
        var head = new AttentionHead(dModel: 3, dHead: 2);

        var query = ReflectionTestHelpers.GetField<Linear>(head, "_query");
        var key = ReflectionTestHelpers.GetField<Linear>(head, "_key");
        var value = ReflectionTestHelpers.GetField<Linear>(head, "_value");

        ReflectionTestHelpers.SetLinearWeights(
            query,
            new double[,] { { 0.1, -0.2 }, { 0.3, 0.4 }, { -0.1, 0.2 } },
            new double[] { 0.05, -0.05 });
        ReflectionTestHelpers.SetLinearWeights(
            key,
            new double[,] { { -0.3, 0.1 }, { 0.2, -0.4 }, { 0.15, 0.25 } },
            new double[] { 0.1, 0.0 });
        ReflectionTestHelpers.SetLinearWeights(
            value,
            new double[,] { { 0.2, 0.3 }, { -0.1, 0.05 }, { 0.4, -0.2 } },
            new double[] { -0.05, 0.05 });

        var input = new double[,] { { 1, -2, 0.5 }, { 0.3, 1.2, -0.7 } };
        var gradOutput = new double[,] { { 1, -1 }, { 0.5, 0.2 } };

        head.Forward(input);
        var (gradInput, gradWq, gradBq, gradWk, gradBk, gradWv, gradBv) = head.Backward(input, gradOutput);

        double Loss(double[,] x) => NumericalGradient.DotProduct(head.Forward(x), gradOutput);

        NumericalGradient.AssertMatrixGradient(input, gradInput, () => Loss(input));

        NumericalGradient.AssertMatrixGradient(ReflectionTestHelpers.GetField<double[,]>(query, "_weights"), gradWq, () => Loss(input));
        NumericalGradient.AssertVectorGradient(ReflectionTestHelpers.GetField<double[]>(query, "_bias"), gradBq, () => Loss(input));

        NumericalGradient.AssertMatrixGradient(ReflectionTestHelpers.GetField<double[,]>(key, "_weights"), gradWk, () => Loss(input));
        NumericalGradient.AssertVectorGradient(ReflectionTestHelpers.GetField<double[]>(key, "_bias"), gradBk, () => Loss(input));

        NumericalGradient.AssertMatrixGradient(ReflectionTestHelpers.GetField<double[,]>(value, "_weights"), gradWv, () => Loss(input));
        NumericalGradient.AssertVectorGradient(ReflectionTestHelpers.GetField<double[]>(value, "_bias"), gradBv, () => Loss(input));
    }
}
