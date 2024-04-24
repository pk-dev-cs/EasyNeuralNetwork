namespace EasyNeuralNetwork.NeuralNetwork
{
    public class Training
    {
        public double Expected { get; }
        public double[] Input { get; }

        public Training(double expected, params double[] input)
        {
            Expected = expected;
            Input = input;
        }
    }
}
