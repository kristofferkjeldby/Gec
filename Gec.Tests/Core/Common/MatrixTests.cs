using Gec.Core.Common;

namespace Gec.Tests.Common;

public class MatrixTests
{
    [Test]
    public void Matadd_AddsElementwise()
    {
        var a = new double[,] { { 1, 2 }, { 3, 4 } };
        var b = new double[,] { { 10, 20 }, { 30, 40 } };

        var result = Matrix.Matadd(a, b);

        AssertMatrixEqual(new double[,] { { 11, 22 }, { 33, 44 } }, result);
    }

    [Test]
    public void Matadd_ThrowsOnRowCountMismatch()
    {
        var a = new double[2, 2];
        var b = new double[3, 2];

        Assert.Throws<Exception>(() => Matrix.Matadd(a, b));
    }

    [Test]
    public void Matadd_ThrowsOnColumnCountMismatch()
    {
        var a = new double[2, 2];
        var b = new double[2, 3];

        Assert.Throws<Exception>(() => Matrix.Matadd(a, b));
    }

    [Test]
    public void ApplyElement_AppliesFunctionToEveryEntry()
    {
        var matrix = new double[,] { { 1, 2 }, { 3, 4 } };

        var result = Matrix.ApplyElement(matrix, x => x * 2);

        AssertMatrixEqual(new double[,] { { 2, 4 }, { 6, 8 } }, result);
    }

    [Test]
    public void ApplyElementBackward_MultipliesGradOutputByDerivativeAtEachEntry()
    {
        var matrix = new double[,] { { 1, 2 }, { 3, 4 } };
        var gradOutput = new double[,] { { 1, 0.5 }, { -1, 2 } };

        var result = Matrix.ApplyElementBackward(matrix, gradOutput, x => x * x);

        // derivative func here is x^2 itself (just a stand-in), so expected = gradOutput * matrix^2 elementwise
        AssertMatrixEqual(new double[,] { { 1, 2 }, { -9, 32 } }, result);
    }

    [Test]
    public void ApplyRows_AppliesFunctionToEachRowIndependently()
    {
        var matrix = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };

        var result = Matrix.ApplyRows(matrix, row => row.Reverse().ToArray());

        AssertMatrixEqual(new double[,] { { 3, 2, 1 }, { 6, 5, 4 } }, result);
    }

    [Test]
    public void ApplyRowsBackward_AppliesFunctionToEachRowPairIndependently()
    {
        var output = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var gradOutput = new double[,] { { 1, 0, -1 }, { 2, -2, 0.5 } };

        var result = Matrix.ApplyRowsBackward(output, gradOutput, (row, gradRow) => row.Select((v, i) => v * gradRow[i]).ToArray());

        AssertMatrixEqual(new double[,] { { 1, 0, -3 }, { 8, -10, 3 } }, result);
    }

    [Test]
    public void ConcatColumns_ConcatenatesMatricesSideBySide()
    {
        var m1 = new double[,] { { 1, 2 }, { 3, 4 } };
        var m2 = new double[,] { { 5 }, { 6 } };
        var m3 = new double[,] { { 7, 8 }, { 9, 10 } };

        var result = Matrix.ConcatColumns(m1, m2, m3);

        AssertMatrixEqual(new double[,] { { 1, 2, 5, 7, 8 }, { 3, 4, 6, 9, 10 } }, result);
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
