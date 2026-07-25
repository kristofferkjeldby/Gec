namespace Gec.Tokenizer.Models;

using System.Collections.Generic;

public class MergeRule
{
    private readonly string rule;
    private readonly int firstToken;
    private readonly int secondToken;
    private readonly int mergedToken;

    public MergeRule(string text, IDictionary<string, int> vocabulary)
    {
        rule = text;

        var split = text.Split(new char[] { Constants.Space });

        firstToken = vocabulary[split[0]];
        secondToken = vocabulary[split[1]];
        mergedToken = vocabulary[rule.Replace(Constants.SpaceString, string.Empty)];
    }

    public bool Apply(List<int> tokens)
    {
        var changed = false;
        var result = new List<int>(tokens.Count);

        for (var i = 0; i < tokens.Count; i++)
        {
            if (i < tokens.Count - 1 && tokens[i] == firstToken && tokens[i + 1] == secondToken)
            {
                result.Add(mergedToken);
                i++;
                changed = true;
            }
            else
            {
                result.Add(tokens[i]);
            }
        }

        if (changed)
        {
            tokens.Clear();
            tokens.AddRange(result);
        }

        return changed;
    }

    public override string ToString() => rule;
}