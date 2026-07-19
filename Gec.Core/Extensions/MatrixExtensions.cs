namespace Gec.Core.Extensions;

public static class MatrixExtensions
{
    public static int Rows(this double[,] matrix)
    {
        return matrix.GetLength(0);
    }
    
    public static int Cols(this double[,] matrix)
    {
        return matrix.GetLength(1);
    }
    
    public static double[] Row(this double[,] matrix, int row)
    {
        var cols = matrix.Cols();
        var result = new double[cols];

        for (int j = 0; j < cols; j++)
        {
            result[j] = matrix[row, j];
        }

        return result;
    }

    
    public static double[,] Transpose(this double[,] matrix)
    {
        var rows = matrix.Rows();
        var cols = matrix.Cols();
        
        var result = new double[cols, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[j, i] = matrix[i, j];   
            }
        }
        
        return result;
    }
}