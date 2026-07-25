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

    [TestCase(1, 1, 1)]
    [TestCase(3, 1, 4)]
    [TestCase(7, 5, 3)]
    [TestCase(1, 9, 1)]
    [TestCase(16, 16, 16)]
    [TestCase(17, 23, 19)]
    [TestCase(64, 96, 129)]
    public void Forward_MatchesNaiveReference(int rows, int inner, int cols)
    {
        var random = new Random(rows * 1000 + inner * 10 + cols);
        var a = RandomMatrix(random, rows, inner);
        var b = RandomMatrix(random, inner, cols);

        AssertMatrixEqual(NaiveProduct(a, b), Matmul.Forward(a, b), 1e-9);
    }

    [Test]
    public void Forward_HandlesZeroSizedOperands()
    {
        Assert.That(Matmul.Forward(new double[0, 3], new double[3, 4]).GetLength(0), Is.EqualTo(0));
        Assert.That(Matmul.Forward(new double[2, 0], new double[0, 4]), Is.EqualTo(new double[2, 4]));
        Assert.That(Matmul.Forward(new double[2, 3], new double[3, 0]).GetLength(1), Is.EqualTo(0));
    }

    private static double[,] RandomMatrix(Random random, int rows, int cols)
    {
        var matrix = new double[rows, cols];

        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            matrix[i, j] = random.NextDouble() * 4 - 2;

        return matrix;
    }

    private static double[,] NaiveProduct(double[,] a, double[,] b)
    {
        var rows = a.GetLength(0);
        var inner = a.GetLength(1);
        var cols = b.GetLength(1);

        var result = new double[rows, cols];

        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
        {
            double sum = 0;
            for (var k = 0; k < inner; k++)
                sum += a[i, k] * b[k, j];
            result[i, j] = sum;
        }

        return result;
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
