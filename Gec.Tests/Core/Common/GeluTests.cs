using Gec.Core.Common;

namespace Gec.Tests.Common;

public class GeluTests
{
    private const double Tolerance = 1e-6;

    [Test]
    public void GeluApproxForward_AtZero_ReturnsZero()
    {
        Assert.That(Gelu.GeluApproxForward(0), Is.EqualTo(0).Within(Tolerance));
    }

    [Test]
    public void GeluApproxForward_MatchesKnownReferenceValues()
    {
        // Reference values for the tanh-approximation of GELU at x = 1 and x = -1.
        Assert.That(Gelu.GeluApproxForward(1), Is.EqualTo(0.8412).Within(1e-3));
        Assert.That(Gelu.GeluApproxForward(-1), Is.EqualTo(-0.1588).Within(1e-3));
    }

    [Test]
    public void GeluApproxForward_LargePositiveValue_ApproachesIdentity()
    {
        Assert.That(Gelu.GeluApproxForward(10), Is.EqualTo(10).Within(1e-3));
    }

    [Test]
    public void GeluApproxForward_LargeNegativeValue_ApproachesZero()
    {
        Assert.That(Gelu.GeluApproxForward(-10), Is.EqualTo(0).Within(1e-3));
    }

    [TestCase(-2.0)]
    [TestCase(-1.0)]
    [TestCase(-0.5)]
    [TestCase(0.0)]
    [TestCase(0.5)]
    [TestCase(1.0)]
    [TestCase(2.0)]
    public void GeluApproxDerivative_MatchesCentralDifferenceOfForward(double x)
    {
        const double h = 1e-5;
        var numericalDerivative = (Gelu.GeluApproxForward(x + h) - Gelu.GeluApproxForward(x - h)) / (2 * h);

        var analyticalDerivative = Gelu.GeluApproxDerivative(x);

        Assert.That(analyticalDerivative, Is.EqualTo(numericalDerivative).Within(1e-4));
    }
}
