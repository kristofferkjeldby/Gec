using Gec.Core.Extensions;

namespace Gec.Tests.Extensions;

public class MatrixExtensionsTests
{
    [Test]
    public void Rows_ReturnsRowCount()
    {
        var matrix = new double[3, 5];

        Assert.That(matrix.Rows(), Is.EqualTo(3));
    }

    [Test]
    public void Cols_ReturnsColumnCount()
    {
        var matrix = new double[3, 5];

        Assert.That(matrix.Cols(), Is.EqualTo(5));
    }

    [Test]
    public void Row_ExtractsRequestedRow()
    {
        var matrix = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };

        var result = matrix.Row(1);

        Assert.That(result, Is.EqualTo(new double[] { 4, 5, 6 }).Within(1e-9));
    }

    [Test]
    public void Transpose_SwapsRowsAndColumns()
    {
        var matrix = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };

        var result = matrix.Transpose();

        var expected = new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } };
        Assert.That(result.GetLength(0), Is.EqualTo(expected.GetLength(0)));
        Assert.That(result.GetLength(1), Is.EqualTo(expected.GetLength(1)));

        for (var i = 0; i < expected.GetLength(0); i++)
        for (var j = 0; j < expected.GetLength(1); j++)
            Assert.That(result[i, j], Is.EqualTo(expected[i, j]).Within(1e-9), $"Mismatch at [{i},{j}]");
    }
}
