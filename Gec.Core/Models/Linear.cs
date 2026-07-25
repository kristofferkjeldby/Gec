using Gec.Core.Common;
using Gec.Core.Extensions;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class Linear
{
    private readonly double[,] _weights; // shape: [inputDim, outputDim]
    private readonly double[] _bias;     // shape: [outputDim]

    private readonly MatrixParameter _weightParameter;
    private readonly VectorParameter _biasParameter;

    private double[,] _input = null!;

    public Linear(int inputDim, int outputDim, Random? random = null, string name = "linear")
    {
        _weights = new double[inputDim, outputDim];
        _bias = new double[outputDim];

        random ??= new Random();
        for (var i = 0; i < inputDim; i++)
        {
            for (var j = 0; j < outputDim; j++)
            {
                _weights[i, j] = random.NextGaussian(0.02);
            }
        }

        _weightParameter = new MatrixParameter($"{name}.weights", _weights);
        _biasParameter = new VectorParameter($"{name}.bias", _bias);
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return _weightParameter;
        yield return _biasParameter;
    }

    public double[,] Forward(double[,] input) // input shape: [seqLen, inputDim]
    {
        _input = input;

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

    public double[,] Backpropagate(double[,] gradOutput)
    {
        var (gradInput, gradWeights, gradBias) = Backward(_input, gradOutput);

        _weightParameter.AddGradient(gradWeights);
        _biasParameter.AddGradient(gradBias);

        return gradInput;
    }
}
