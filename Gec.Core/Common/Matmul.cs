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

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double r = 0;

                for (int k = 0; k < inner; k++)
                {
                    r += matrix1[i, k] * matrix2[k, j];
                }

                result[i, j] = r;
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