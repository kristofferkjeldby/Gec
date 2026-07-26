using Gec.Core.Extensions;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class LayerNorm
{
    private readonly double[] _gamma;
    private readonly double[] _beta;
    private readonly double _epsilon;

    private readonly VectorParameter _gammaParameter;
    private readonly VectorParameter _betaParameter;

    // Populated by Forward and read by Backward; only valid after a Forward call.
    private double[,] _xHat = null!;
    private double[] _stdInv = null!;

    public LayerNorm(int dimension, double epsilon = 1e-5, string name = "layerNorm")
    {
        _gamma = Enumerable.Repeat(1.0, dimension).ToArray();
        _beta = new double[dimension];
        _epsilon = epsilon;

        _gammaParameter = new VectorParameter($"{name}.gamma", _gamma);
        _betaParameter = new VectorParameter($"{name}.beta", _beta);
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return _gammaParameter;
        yield return _betaParameter;
    }

    /// <summary>
    /// Apply layer normalization to each row in a matrix. Layer normalization takes the elements in a row and shift
    /// them so the average is zero, and scale them so the variance is 1. It then takes two learned parameters
    /// and scales/shifts the normalized values using gamma/beta.
    /// </summary>
    public double[,] Forward(double[,] input) // input shape: [seqLen, dim]
    {
        var rows = input.Rows();
        var cols = input.Cols();

        var output = new double[rows, cols];
        _xHat = new double[rows, cols];
        _stdInv = new double[rows];

        for (var r = 0; r < rows; r++)
        {
            var row = input.Row(r);
            var average = row.Average();
            var variance = row.Average(e => (e - average) * (e - average));
            var stdInv = 1.0 / Math.Sqrt(variance + _epsilon);
            _stdInv[r] = stdInv;

            for (var c = 0; c < cols; c++)
            {
                var xHat = (row[c] - average) * stdInv;
                _xHat[r, c] = xHat;
                output[r, c] = _gamma[c] * xHat + _beta[c];
            }
        }

        return output;
    }

    public (double[,] gradInput, double[] gradGamma, double[] gradBeta) Backward(double[,] gradOutput)
    {
        var rows = gradOutput.Rows();
        var cols = gradOutput.Cols();

        var gradInput = new double[rows, cols];
        var gradGamma = new double[cols];
        var gradBeta = new double[cols];

        for (var r = 0; r < rows; r++)
        {
            var g = gradOutput.Row(r);
            var xHat = _xHat.Row(r);
            var stdInv = _stdInv[r];

            var dxHat = new double[cols];
            for (var c = 0; c < cols; c++)
            {
                dxHat[c] = g[c] * _gamma[c];
                gradGamma[c] += g[c] * xHat[c];
                gradBeta[c] += g[c];
            }

            var meanDxHat = dxHat.Average();
            var meanDxHatXHat = dxHat.Select((d, c) => d * xHat[c]).Average();

            for (var c = 0; c < cols; c++)
            {
                gradInput[r, c] = stdInv * (dxHat[c] - meanDxHat - xHat[c] * meanDxHatXHat);
            }
        }

        return (gradInput, gradGamma, gradBeta);
    }

    public double[,] Backpropagate(double[,] gradOutput)
    {
        var (gradInput, gradGamma, gradBeta) = Backward(gradOutput);

        _gammaParameter.AddGradient(gradGamma);
        _betaParameter.AddGradient(gradBeta);

        return gradInput;
    }
}