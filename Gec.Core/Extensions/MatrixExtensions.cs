namespace Gec.Core.Extensions;

/// <summary>
/// Collection of matrix extensions
/// </summary>
public static class MatrixExtensions
{
    /// <summary>
    /// Returns the number of rows in a matrix
    /// </summary>
    public static int Rows(this double[,] matrix)
    {
        return matrix.GetLength(0);
    }
    
    /// <summary>
    /// Returns the number of columns in a matrix
    /// </summary>
    public static int Cols(this double[,] matrix)
    {
        return matrix.GetLength(1);
    }
    
    /// <summary>
    /// Returns a specific row in a matrix as a vector
    /// </summary>
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

    /// <summary>
    /// Transposes a matrix
    /// </summary>
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