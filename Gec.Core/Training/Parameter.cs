namespace Gec.Core.Training;

public abstract class Parameter
{
    protected Parameter(string name, int rows, int cols)
    {
        Name = name;
        Rows = rows;
        Cols = cols;
    }

    public string Name { get; }

    public int Rows { get; }

    public int Cols { get; }

    public int Count => Rows * Cols;

    public abstract double GetValue(int index);

    public abstract void SetValue(int index, double value);

    public abstract double GetGradient(int index);

    public abstract void ZeroGradient();

    public abstract void AddGradientFrom(Parameter other);

    public abstract void CopyValuesFrom(Parameter other);
}

public sealed class MatrixParameter : Parameter
{
    private readonly double[,] _value;

    public MatrixParameter(string name, double[,] value)
        : base(name, value.GetLength(0), value.GetLength(1))
    {
        _value = value;
        Gradient = new double[Rows, Cols];
    }

    public double[,] Gradient { get; }

    public override double GetValue(int index) => _value[index / Cols, index % Cols];

    public override void SetValue(int index, double value) => _value[index / Cols, index % Cols] = value;

    public override double GetGradient(int index) => Gradient[index / Cols, index % Cols];

    public override void ZeroGradient() => Array.Clear(Gradient);

    public void AddGradient(double[,] gradient)
    {
        for (var i = 0; i < Rows; i++)
        for (var j = 0; j < Cols; j++)
            Gradient[i, j] += gradient[i, j];
    }

    public void AddRowGradient(int row, double[] gradient)
    {
        for (var j = 0; j < Cols; j++)
            Gradient[row, j] += gradient[j];
    }

    public override void AddGradientFrom(Parameter other)
    {
        AddGradient(((MatrixParameter)other).Gradient);
    }

    public override void CopyValuesFrom(Parameter other)
    {
        Array.Copy(((MatrixParameter)other)._value, _value, _value.Length);
    }
}

public sealed class VectorParameter : Parameter
{
    private readonly double[] _value;

    public VectorParameter(string name, double[] value)
        : base(name, 1, value.Length)
    {
        _value = value;
        Gradient = new double[value.Length];
    }

    public double[] Gradient { get; }

    public override double GetValue(int index) => _value[index];

    public override void SetValue(int index, double value) => _value[index] = value;

    public override double GetGradient(int index) => Gradient[index];

    public override void ZeroGradient() => Array.Clear(Gradient);

    public void AddGradient(double[] gradient)
    {
        for (var i = 0; i < Gradient.Length; i++)
            Gradient[i] += gradient[i];
    }

    public override void AddGradientFrom(Parameter other)
    {
        AddGradient(((VectorParameter)other).Gradient);
    }

    public override void CopyValuesFrom(Parameter other)
    {
        Array.Copy(((VectorParameter)other)._value, _value, _value.Length);
    }
}
