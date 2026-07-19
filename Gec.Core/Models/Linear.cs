using Gec.Core.Common;

namespace Gec.Core.Models;

public class Linear
{
    private readonly double[,] _weights; // shape: [inputDim, outputDim]
    private readonly double[] _bias;     // shape: [outputDim]
    private readonly Random _random = new Random();
    
    public Linear(int inputDim, int outputDim)
    {
        _weights = new double[inputDim, outputDim];
        _bias = new double[outputDim];
        
        for (var i = 0; i < inputDim; i++)
        {
            for (var j = 0; j < outputDim; j++)
            {
                _weights[i, j] = _random.Next(-100, 100) / 1000d;
            }
        }  
    }

    public double[,] Forward(double[,] input) // input shape: [seqLen, inputDim]
    {
        var w = Matmul.Forward(input, _weights);
        
        for (var i = 0; i < w.GetLength(0); i++)
        {
            for (var j = 0; j < w.GetLength(1); j++)
            {
                w[i, j] +=  _bias[j];
            }
        }  
        
        return w;
    }
    
    public (double[,] gradInput, double[,] gradWeights, double[] gradBias) Backward(double[,] input, double[,] gradOutput)
    {
        var (gradInput, gradWeights) = Matmul.Backward(input, _weights, gradOutput);
        var gradBias = Enumerable.Range(0, gradOutput.GetLength(1))
            .Select(j => Enumerable.Range(0, gradOutput.GetLength(0)).Sum(i => gradOutput[i, j])).ToArray();

        return (gradInput, gradWeights, gradBias);
    }
}

