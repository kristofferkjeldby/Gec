using System.Numerics;
using System.Runtime.InteropServices;
using Gec.Core.Extensions;

namespace Gec.Core.Common;

public static class Matmul
{
    public static double[,] Forward(double[,] matrix1, double[,] matrix2)
    {
        if (matrix1.Cols() != matrix2.Rows())
            throw new Exception("Inner dimensions does not match");

        int rows = matrix1.Rows();
        int inner = matrix1.Cols();
        int cols = matrix2.Cols();

        var result = new double[rows, cols];

        if (rows == 0 || inner == 0 || cols == 0)
            return result;

        var left = MemoryMarshal.CreateSpan(ref matrix1[0, 0], matrix1.Length);
        var right = MemoryMarshal.CreateSpan(ref matrix2[0, 0], matrix2.Length);
        var target = MemoryMarshal.CreateSpan(ref result[0, 0], result.Length);

        var width = Vector<double>.Count;

        for (var i = 0; i < rows; i++)
        {
            var resultRow = target.Slice(i * cols, cols);

            for (var k = 0; k < inner; k++)
            {
                var scale = left[i * inner + k];

                if (scale == 0)
                    continue;

                var sourceRow = right.Slice(k * cols, cols);
                var scaleVector = new Vector<double>(scale);

                var j = 0;

                for (; j <= cols - width; j += width)
                {
                    var updated = new Vector<double>(sourceRow.Slice(j, width)) * scaleVector
                                  + new Vector<double>(resultRow.Slice(j, width));

                    updated.CopyTo(resultRow.Slice(j, width));
                }

                for (; j < cols; j++)
                    resultRow[j] += scale * sourceRow[j];
            }
        }

        return result;
    }

    public static (double[,] gradA, double[,] gradB) Backward(double[,] a, double[,] b, double[,] gradOutput)
    {
        var gradA = Forward(gradOutput, b.Transpose());
        var gradB = Forward(a.Transpose(), gradOutput);

        return (gradA, gradB);
    }
}
