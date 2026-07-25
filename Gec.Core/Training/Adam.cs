namespace Gec.Core.Training;

public sealed class Adam
{
    private readonly Parameter[] _parameters;
    private readonly double[][] _firstMoment;
    private readonly double[][] _secondMoment;
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    private int _step;

    public Adam(IEnumerable<Parameter> parameters, double learningRate = 3e-4, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        _parameters = parameters.ToArray();
        _firstMoment = _parameters.Select(p => new double[p.Count]).ToArray();
        _secondMoment = _parameters.Select(p => new double[p.Count]).ToArray();
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    public void ZeroGradients()
    {
        foreach (var parameter in _parameters)
            parameter.ZeroGradient();
    }

    public void ClipGradients(double maxNorm)
    {
        double sumOfSquares = 0;
        foreach (var parameter in _parameters)
        for (var i = 0; i < parameter.Count; i++)
        {
            var gradient = parameter.GetGradient(i);
            sumOfSquares += gradient * gradient;
        }

        var norm = Math.Sqrt(sumOfSquares);
        if (norm <= maxNorm || norm == 0)
            return;

        var scale = maxNorm / norm;
        foreach (var parameter in _parameters)
        {
            for (var i = 0; i < parameter.Count; i++)
                ScaleGradient(parameter, i, scale);
        }
    }

    public void Step()
    {
        _step++;

        var firstCorrection = 1 - Math.Pow(_beta1, _step);
        var secondCorrection = 1 - Math.Pow(_beta2, _step);

        for (var p = 0; p < _parameters.Length; p++)
        {
            var parameter = _parameters[p];
            var firstMoment = _firstMoment[p];
            var secondMoment = _secondMoment[p];

            for (var i = 0; i < parameter.Count; i++)
            {
                var gradient = parameter.GetGradient(i);

                firstMoment[i] = _beta1 * firstMoment[i] + (1 - _beta1) * gradient;
                secondMoment[i] = _beta2 * secondMoment[i] + (1 - _beta2) * gradient * gradient;

                var step = _learningRate * (firstMoment[i] / firstCorrection)
                           / (Math.Sqrt(secondMoment[i] / secondCorrection) + _epsilon);

                parameter.SetValue(i, parameter.GetValue(i) - step);
            }
        }
    }

    private static void ScaleGradient(Parameter parameter, int index, double scale)
    {
        switch (parameter)
        {
            case MatrixParameter matrix:
                matrix.Gradient[index / matrix.Cols, index % matrix.Cols] *= scale;
                break;
            case VectorParameter vector:
                vector.Gradient[index] *= scale;
                break;
        }
    }
}
