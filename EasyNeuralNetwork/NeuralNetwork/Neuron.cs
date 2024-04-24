namespace EasyNeuralNetwork.NeuralNetwork;

public class Neuron
{
    public Neuron(int layerIndex)
    {
        LayerIndex = layerIndex;
        Guid = Guid.NewGuid();
    }

    public double LayerIndex { get; }
    public Guid Guid { get; }

    public double Input { get; set; }
    public double Output { get; set; }
    public double Bias { get; set; } = 0.0;
    public double Error { get; set; }
}
