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

        var count = tokens.Count;

        for (int offset = 0; offset < count-1; offset++)
        {
            var currentFirstToken = tokens[offset];
            var currentSecondToken = tokens[offset+1];

            if (currentFirstToken == firstToken && currentSecondToken == secondToken)
            {
                changed = true;
                tokens[offset] = mergedToken;
                tokens.RemoveAt(offset + 1);
                offset++;
                count--;
            }
        }

        return changed;

    }

    public override string ToString() => rule;
}