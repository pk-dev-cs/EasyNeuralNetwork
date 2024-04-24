namespace EasyNeuralNetwork.NeuralNetwork;

public class Layer
{
    public int LayerIndex { get; }

    public List<Neuron> Neurons = [];
    public List<Synapse> Synapses = [];

    public Layer(int neuronsCount, int layerIndex)
    {
        LayerIndex = layerIndex;
        for (int i = 0; i < neuronsCount; i++)
            Neurons.Add(new Neuron(layerIndex));
    }
}
