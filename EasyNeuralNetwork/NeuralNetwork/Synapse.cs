namespace EasyNeuralNetwork.NeuralNetwork;

public class Synapse
{
    public Neuron From { get; }
    public Neuron To { get; }
    public double Weight { get; set; }

    public Synapse(Neuron from, Neuron to, double weight)
    {
        From = from;
        To = to;
        Weight = weight;
    }
}
