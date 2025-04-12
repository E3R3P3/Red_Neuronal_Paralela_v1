using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Red_Neuronal_Paralela_v1
{
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
        public readonly int inputSize;
        public readonly int hiddenSize;
        public readonly double learningRate;
        public double[][] hiddenWeights;
        public double[] outputWeights;
        public readonly Random rand;

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

        public double DotProduct(double[] a, double[] b)
        {
            double result = 0;
            for (int i = 0; i < a.Length; i++)
                result += a[i] * b[i];
            return result;
        }

        public double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        internal void SaveModel(string modelPath)
        {
            // Crear un objeto con los pesos que queremos guardar
            var networkData = new NeuralNetworkData
            {
                HiddenWeights = this.hiddenWeights,
                OutputWeights = this.outputWeights,
                InputSize = this.inputSize,
                HiddenSize = this.hiddenSize
            };

            // Serializar a JSON y guardar en el archivo
            var jsonData = JsonSerializer.Serialize(networkData);
            File.WriteAllText(modelPath, jsonData);

            Console.WriteLine($"Red neuronal guardada en: {modelPath}");
        }

        internal static NeuralNetwork LoadModel(string modelPath)
        {
            // Leer el archivo JSON
            var jsonData = File.ReadAllText(modelPath);
            // Deserializar el JSON a un objeto
            var networkData = JsonSerializer.Deserialize<NeuralNetworkData>(jsonData);
            // Crear una nueva instancia de NeuralNetwork con los pesos cargados
            var nn = new NeuralNetwork(inputSize: networkData.InputSize, hiddenSize: networkData.HiddenSize);
            nn.hiddenWeights = networkData.HiddenWeights;
            nn.outputWeights = networkData.OutputWeights;
            return nn;
        }
    }
}

