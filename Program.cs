using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

// Paso 1: Preprocesamiento de datos
public class DataLoader
{
    public List<DataPoint> LoadData(string filePath)
    {
        var data = new List<DataPoint>();
        var lines = File.ReadAllLines(filePath).Skip(1);
        
        foreach (var line in lines)
        {
            var parts = line.Split(',');
            if (parts.Length < 4) continue;
            
            data.Add(new DataPoint(
                Position: parts[1].Trim(),
                Gender: parts[2].Trim().Equals("Masculino", StringComparison.OrdinalIgnoreCase) ? 0 : 1,
                Salary: double.Parse(parts[3].Trim(), CultureInfo.InvariantCulture)
            ));
        }
        return data;
    }

    public (double[][], double[]) PrepareData(List<DataPoint> data)
    {
        // Obtener los 10 puestos más comunes
        var positions = data
            .GroupBy(d => d.Position)
            .OrderByDescending(g => g.Count())
            .Take(10)
            .Select(g => g.Key)
            .ToList();
        
        var inputs = new List<double[]>();
        var outputs = new List<double>();
        
        // Normalización de salarios
        double maxSalary = data.Max(d => d.Salary);
        double minSalary = data.Min(d => d.Salary);
        
        foreach (var item in data)
        {
            var encoded = new List<double> { item.Gender };
            foreach (var pos in positions)
                encoded.Add(item.Position == pos ? 1 : 0);
            
            inputs.Add(encoded.ToArray());
            outputs.Add((item.Salary - minSalary) / (maxSalary - minSalary)); // Normalización Min-Max
        }
        return (inputs.ToArray(), outputs.ToArray());
    }
}

public record DataPoint(string Position, int Gender, double Salary);

// Paso 2: Implementación de la red neuronal
public class NeuralNetwork
{
    private readonly int inputSize;
    private readonly int hiddenSize;
    private readonly double learningRate;
    private double[][] hiddenWeights;
    private double[] outputWeights;
    private readonly Random rand;

    public NeuralNetwork(int inputSize, int hiddenSize, double learningRate = 0.01)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.learningRate = learningRate;
        this.rand = new Random(Guid.NewGuid().GetHashCode());
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        hiddenWeights = new double[hiddenSize][];
        for (int i = 0; i < hiddenSize; i++)
        {
            hiddenWeights[i] = new double[inputSize];
            for (int j = 0; j < inputSize; j++)
                hiddenWeights[i][j] = rand.NextDouble() * 2 - 1;
        }
        
        outputWeights = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
            outputWeights[i] = rand.NextDouble() * 2 - 1;
    }

    public double Forward(double[] input, bool parallel = false)
    {
        var hiddenOutputs = new double[hiddenSize];
        
        if (parallel)
        {
            Parallel.For(0, hiddenSize, i => 
            {
                hiddenOutputs[i] = Sigmoid(DotProduct(input, hiddenWeights[i]));
            });
        }
        else
        {
            for (int i = 0; i < hiddenSize; i++)
                hiddenOutputs[i] = Sigmoid(DotProduct(input, hiddenWeights[i]));
        }
        
        return Sigmoid(DotProduct(hiddenOutputs, outputWeights));
    }

    private double DotProduct(double[] a, double[] b)
    {
        double result = 0;
        for (int i = 0; i < a.Length; i++)
            result += a[i] * b[i];
        return result;
    }

    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
}

// Paso 3: Entrenamiento con paralelismo
public class Trainer
{
    public void Train(NeuralNetwork nn, double[][] inputs, double[] targets, int epochs, bool parallel = true)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            object lockObj = new object();
            
            if (parallel)
            {
                Parallel.For(0, inputs.Length, i =>
                {
                    var error = TrainSample(nn, inputs[i], targets[i]);
                    lock (lockObj) totalError += error;
                });
            }
            else
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    totalError += TrainSample(nn, inputs[i], targets[i]);
                }
            }
            
            if (epoch % 10 == 0)
                Console.WriteLine($"Epoch {epoch}, Error: {totalError / inputs.Length}");
        }
    }

    private double TrainSample(NeuralNetwork nn, double[] input, double target)
    {
        // Implementación básica de retropropagación
        // (Este es un ejemplo simplificado, necesitarías implementar la lógica completa)
        double output = nn.Forward(input);
        double error = Math.Pow(output - target, 2);
        // Aquí iría la actualización de pesos
        return error;
    }
}

// Paso 4: Uso principal
class Program
{
    static void Main()
    {
        var loader = new DataLoader();
        var data = loader.LoadData("nomina_hacienda.csv");
        var (inputs, targets) = loader.PrepareData(data);
        
        var nn = new NeuralNetwork(inputSize: inputs[0].Length, hiddenSize: 5);
        var trainer = new Trainer();
        
        // Entrenamiento secuencial
        var watchSeq = System.Diagnostics.Stopwatch.StartNew();
        trainer.Train(nn, inputs, targets, epochs: 100, parallel: false);
        watchSeq.Stop();
        
        // Entrenamiento paralelo
        var watchPar = System.Diagnostics.Stopwatch.StartNew();
        trainer.Train(nn, inputs, targets, epochs: 100, parallel: true);
        watchPar.Stop();
        
        Console.WriteLine("Resultados:");
        Console.WriteLine($"Tiempo secuencial: {watchSeq.ElapsedMilliseconds}ms");
        Console.WriteLine($"Tiempo paralelo: {watchPar.ElapsedMilliseconds}ms");
    }
}