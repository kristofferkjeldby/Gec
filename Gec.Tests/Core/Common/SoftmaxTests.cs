using Gec.Core.Common;
using Gec.Tests;

namespace Gec.Tests.Common;

public class SoftmaxTests
{
    private const double Tolerance = 1e-6;

    [Test]
    public void Forward_MatchesHandComputedValues()
    {
        var result = Softmax.Forward(new double[] { 1, 2, 3 });

        AssertArrayEqual(new[] { 0.09003057, 0.24472847, 0.66524096 }, result, 1e-6);
    }

    [Test]
    public void Forward_UniformInput_ReturnsUniformDistribution()
    {
        var result = Softmax.Forward(new double[] { 5, 5, 5, 5 });

        AssertArrayEqual(new[] { 0.25, 0.25, 0.25, 0.25 }, result, Tolerance);
    }

    [Test]
    public void Forward_SumsToOne()
    {
        var result = Softmax.Forward(new double[] { -3, 0.5, 2, 10 });

        Assert.That(result.Sum(), Is.EqualTo(1.0).Within(Tolerance));
    }

    [Test]
    public void Forward_IsNumericallyStableForLargeValues()
    {
        var result = Softmax.Forward(new double[] { 1000, 1000, 1000 });

        Assert.That(result, Has.All.Not.NaN);
        AssertArrayEqual(new[] { 1.0 / 3, 1.0 / 3, 1.0 / 3 }, result, Tolerance);
    }

    [Test]
    public void Backward_MatchesNumericalGradientOfForward()
    {
        var input = new double[] { 1, 2, 3 };
        var gradOutput = new double[] { 0.5, -1, 2 };
        var output = Softmax.Forward(input);

        var analytical = Softmax.Backward(output, gradOutput);

        const double h = 1e-6;
        for (var i = 0; i < input.Length; i++)
        {
            var plus = (double[])input.Clone();
            plus[i] += h;
            var minus = (double[])input.Clone();
            minus[i] -= h;

            var dotPlus = Dot(Softmax.Forward(plus), gradOutput);
            var dotMinus = Dot(Softmax.Forward(minus), gradOutput);

            var expected = (dotPlus - dotMinus) / (2 * h);
            Assert.That(analytical[i], Is.EqualTo(expected).Within(1e-4), $"Mismatch at [{i}]");
        }
    }

    private static double Dot(double[] a, double[] b) => a.Select((x, i) => x * b[i]).Sum();

    private static void AssertArrayEqual(double[] expected, double[] actual, double tolerance)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length));

        for (var i = 0; i < expected.Length; i++)
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(tolerance), $"Mismatch at [{i}]");
    }
}
