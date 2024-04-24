using EasyNeuralNetwork.NeuralNetwork;

var network = new NeuralNetwork { Iterations = 10000, Alpha = 3.5, L2_Regularization = true, Rnd = new Random(12345) };
network.AddLayer(2);
network.AddLayer(2);
network.AddLayer(1);

network.CreateSynapses();

var training = new List<Training>
{
    new (0.2, 0.1200, 0.350000),
    new (0.3, 0.1400, 0.380000),
    new (0.3, 0.1600, 0.400000),
    new (0.3, 0.1800, 0.420000),
    new (0.4, 0.2000, 0.450000),
    new (0.4, 0.2200, 0.470000),
    new (0.4, 0.2400, 0.500000),
    new (0.4, 0.2600, 0.520000),
    new (0.4, 0.2800, 0.550000),
    new (0.5, 0.3000, 0.580000),
    new (0.5, 0.3200, 0.600000),
    new (0.5, 0.3400, 0.630000),
    new (0.5, 0.3600, 0.650000),
    new (0.5, 0.3800, 0.680000),
    new (0.6, 0.4000, 0.700000),
    new (0.6, 0.4200, 0.730000),
    new (0.6, 0.4400, 0.750000),
    new (0.6, 0.4600, 0.780000),
    new (0.6, 0.4800, 0.800000),
    new (0.6, 0.5000, 0.830000),
    new (0.7, 0.5200, 0.850000),
    new (0.7, 0.5400, 0.880000),
    new (0.7, 0.5600, 0.900000),
    new (0.7, 0.5800, 0.930000),
    new (0.7, 0.6000, 0.950000),
    new (0.7, 0.6200, 0.980000),
    new (0.8, 0.6400, 1)
};

network.Train(training);

var prediction = network.Predict(new[] { 0.4200, 0.730000 })[0];
Console.WriteLine($"Prediction result: {prediction}");
Console.ReadKey();