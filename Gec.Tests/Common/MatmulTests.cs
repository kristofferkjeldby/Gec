using Gec.Core.Common;

namespace Gec.Tests.Common;

public class MatmulTests
{
    [Test]
    public void Forward_ComputesCorrectProduct()
    {
        var a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var b = new double[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } };

        var result = Matmul.Forward(a, b);

        AssertMatrixEqual(new double[,] { { 58, 64 }, { 139, 154 } }, result);
    }

    [Test]
    public void Forward_ThrowsOnInnerDimensionMismatch()
    {
        var a = new double[2, 3];
        var b = new double[2, 2];

        Assert.Throws<Exception>(() => Matmul.Forward(a, b));
    }

    [Test]
    public void Backwards_ComputesGradientsAsMatmulWithTransposes()
    {
        var a = new double[,] { { 1, 2 }, { 3, 4 } };
        var b = new double[,] { { 5, 6 }, { 7, 8 } };
        var gradOutput = new double[,] { { 1, 0 }, { 0, 1 } };

        var (gradA, gradB) = Matmul.Backward(a, b, gradOutput);

        // gradOutput is the identity, so gradA == b^T and gradB == a^T
        AssertMatrixEqual(new double[,] { { 5, 7 }, { 6, 8 } }, gradA);
        AssertMatrixEqual(new double[,] { { 1, 3 }, { 2, 4 } }, gradB);
    }

    private static void AssertMatrixEqual(double[,] expected, double[,] actual, double tolerance = 1e-6)
    {
        Assert.That(actual.GetLength(0), Is.EqualTo(expected.GetLength(0)), "Row count mismatch");
        Assert.That(actual.GetLength(1), Is.EqualTo(expected.GetLength(1)), "Column count mismatch");

        for (var i = 0; i < expected.GetLength(0); i++)
        for (var j = 0; j < expected.GetLength(1); j++)
            Assert.That(actual[i, j], Is.EqualTo(expected[i, j]).Within(tolerance), $"Mismatch at [{i},{j}]");
    }
}
