using Gec.Core.Common;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class Mlp
{
    private readonly Linear _up;
    private readonly Linear _down;
     
    // Populated by Forward and read by Backward; only valid after a Forward call.
    private double[,] _hidden = null!;
    private double[,] _activated = null!;

    public Mlp(int dModel, int dFf, Random? random = null, string name = "mlp")
    {
        _up = new Linear(dModel, dFf, random, $"{name}.up");
        _down = new Linear(dFf, dModel, random, $"{name}.down");
    }

    public IEnumerable<Parameter> Parameters() => _up.Parameters().Concat(_down.Parameters());

    public double[,] Forward(double[,] input) // input shape: [seqLen, dModel]
    {
        // Step 1
        _hidden = _up.Forward(input);
        
        // Step 2
        _activated = Matrix.ApplyElement(_hidden, Gelu.GeluApproxForward);
        
        // Step 3
        var output = _down.Forward(_activated);
        
        return output;
    }
    
    public (double[,] gradInput, double[,] gradWUp, double[] gradBUp, double[,] gradWDown, double[] gradBDown)
        Backward(double[,] input, double[,] gradOutput)
    {
        // Undo step 3
        var (gradActivated, gradWDown, gradBDown) = _down.Backward(_activated, gradOutput);
        
        // Undo step 2
        var gradHidden = Matrix.ApplyElementBackward(_hidden, gradActivated, Gelu.GeluApproxBackward);
        
        // Undo step 1
        var (gradInput, gradWUp, gradBUp) = _up.Backward(input, gradHidden);

        return (gradInput, gradWUp, gradBUp, gradWDown, gradBDown);
    }

    public double[,] Backpropagate(double[,] gradOutput)
    {
        var gradActivated = _down.Backpropagate(gradOutput);
        var gradHidden = Matrix.ApplyElementBackward(_hidden, gradActivated, Gelu.GeluApproxBackward);

        return _up.Backpropagate(gradHidden);
    }
}