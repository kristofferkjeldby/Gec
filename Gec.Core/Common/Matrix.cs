using Gec.Core.Extensions;

namespace Gec.Core.Common;

public static class Matrix
{
    public static double[,] Matadd(double[,] matrix1, double[,] matrix2)
    {
        if (matrix1.GetLength(0) != matrix2.GetLength(0))
            throw new Exception("Row count does not match");
        
        if (matrix1.GetLength(1) != matrix2.GetLength(1))
            throw new Exception("Column count does not match");

        int rows = matrix1.GetLength(0);
        int cols = matrix1.GetLength(1);

        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = matrix1[i, j] +  matrix2[i, j]; 
            }
        }

        return result;
    }
    
    public static double[,] ApplyElement(double[,] matrix, Func<double, double> func)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        
        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = func(matrix[i, j]);   
            }
        }
        
        return result;
    }
    
    public static double[,] ApplyElementBackward(double[,] matrix, double[,] gradOutput, Func<double, double> derivative)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        
        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = gradOutput[i, j] * derivative(matrix[i, j]);   
            }
        }
        
        return result;
    }
    
    public static double[,] ApplyRows(double[,] matrix, Func<double[], double[]> func)
    {
        var rows = matrix.Rows();
        var cols = matrix.Cols();
        
        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            var row = func(matrix.Row(i));
            
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = row[j];   
            }
        }
        
        return result;
    }
    
    public static double[,] ApplyRowsBackward(double[,] output, double[,] gradOutput, Func<double[], double[], double[]> backwardFunc)
    {
        var rows = output.Rows();
        var cols = output.Cols();
        
        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            var row = backwardFunc(output.Row(i), gradOutput.Row(i));
            
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = row[j];   
            }
        }
        
        return result;
    }
    
    public static double[,] ConcatColumns(params double[][,] matrices)
    {
        var rows = matrices[0].GetLength(0);
        var cols = matrices.Sum(m => m.GetLength(1));

        var result = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            var row =  matrices.Select(m => m.Row(i)).SelectMany(e => e).ToArray();
            
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = row[j];   
            }
        }
        
        return result;
    }
}