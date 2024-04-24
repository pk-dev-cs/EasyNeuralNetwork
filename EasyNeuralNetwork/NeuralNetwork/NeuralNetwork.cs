namespace EasyNeuralNetwork.NeuralNetwork;

public class NeuralNetwork
{
    public Random Rnd { get; set; } = new Random();

    public int Iterations { get; set; } = 1000;
    public double Alpha { get; set; } = 5.5;
    public bool L2_Regularization { get; set; } = true;
    public double Lambda { get; set; } = 0.00003;


    private List<Layer> Layers = [];

    public void AddLayer(int neuronsCount)
    {
        var layer = new Layer(neuronsCount, Layers.Count());
        Layers.Add(layer);
    }

    public void CreateSynapses()
    {
        for (int i = 1; i < Layers.Count; i++)
        {
            var synapses = from x in Layers[i - 1].Neurons
                           from y in Layers[i].Neurons
                           let inputCount = Layers[i - 1].Neurons.Count
                           let outputCount = Layers[i].Neurons.Count
                           let weight = WeightFunction(inputCount, outputCount)
                           select new Synapse(x, y, weight);

            Layers[i].Synapses = synapses.ToList();
        }
    }

    public void Train(List<Training> trainingSet)
    {
        for (int i = 0; i < Iterations; i++)
        {
            var costs = new List<double>();
            foreach (var trainingEntry in trainingSet)
            {
                var output = Predict(trainingEntry.Input);

                var lastLayer = Layers.Last();
                foreach (var neuron in lastLayer.Neurons)
                {
                    var cost = neuron.Output - trainingEntry.Expected;
                    neuron.Error = cost * SigmoidPrime(neuron.Input + neuron.Bias);
                }

                BackPropagate();

                for (int j = 1; j < Layers.Count(); j++)
                {
                    foreach (var neuron in Layers[j].Neurons)
                        neuron.Bias -= Alpha * neuron.Error;
                }

                for (int j = 1; j < Layers.Count() - 1; j++)
                {
                    foreach (var synapse in Layers[j].Synapses)
                    {
                        synapse.Weight -= Alpha * synapse.From.Output * synapse.To.Error;
                        if (L2_Regularization)
                            synapse.Weight -= Lambda * synapse.Weight;

                    }
                }
            }
        }
    }

    private void BackPropagate()
    {
        for (int i = Layers.Count() - 1; i > 1; i--)
        {
            foreach (var synapse in Layers[i].Synapses)
            {
                var sum = synapse.Weight * synapse.To.Error;
                synapse.From.Error = sum * SigmoidPrime(synapse.From.Input + synapse.From.Bias);
            }
        }
    }

    public List<double> Predict(double[] input)
    {
        var firstLayer = Layers.First();
        for (int i = 0; i < firstLayer.Neurons.Count; i++)
            firstLayer.Neurons[i].Output = input[i];

        for (int i = 1; i < Layers.Count; i++)
        {
            for (int j = 0; j < Layers[i].Neurons.Count; j++)
            {
                double sum = 0;
                for (int k = 0; k < Layers[i - 1].Neurons.Count; k++)
                {
                    var synapse = Layers[i].Synapses.First(x => x.From.Guid.Equals(Layers[i - 1].Neurons[k].Guid) && x.To.Guid.Equals(Layers[i].Neurons[j].Guid));
                    sum += synapse.From.Output * synapse.Weight;
                }

                Layers[i].Neurons[j].Input = sum;
                Layers[i].Neurons[j].Output = Sigmoid(Layers[i].Neurons[j].Input + Layers[i].Neurons[j].Bias);
            }
        }

        var output = new List<double>();
        var outputLayer = Layers.Last();
        foreach (var neuron in outputLayer.Neurons)
            output.Add(neuron.Output);

        return output;
    }

    private double WeightFunction(int inputCount, int outputCount)
    {
        var b = Math.Sqrt(6) / Math.Sqrt(inputCount + outputCount);
        return Rnd.NextDouble() * 2 * b - b;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double SigmoidPrime(double x) => Sigmoid(x) * (1.0 - Sigmoid(x));
}
