using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Red_Neuronal_Paralela_v1
{
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
            // Paso 1: Forward pass
            double[] hiddenOutputs = new double[nn.hiddenSize];
            for (int i = 0; i < nn.hiddenSize; i++)
                hiddenOutputs[i] = nn.Sigmoid(nn.DotProduct(input, nn.hiddenWeights[i]));

            double output = nn.Sigmoid(nn.DotProduct(hiddenOutputs, nn.outputWeights));

            // Paso 2: Calcular el error (Error cuadrático medio)
            double outputError = output - target;
            double sigmoidOutput = nn.Sigmoid(output);
            double sigmoidPrime = sigmoidOutput * (1 - sigmoidOutput);
            double outputGradient = outputError * sigmoidPrime;

            // Paso 3: Actualizar los pesos de la capa de salida
            for (int i = 0; i < nn.hiddenSize; i++)
            {
                nn.outputWeights[i] -= nn.learningRate * outputGradient * hiddenOutputs[i];
            }

            // Paso 4: Retropropagación hacia la capa oculta
            double[] hiddenErrors = new double[nn.hiddenSize];
            for (int i = 0; i < nn.hiddenSize; i++)
            {
                hiddenErrors[i] = outputGradient * nn.outputWeights[i] * SigmoidPrime(hiddenOutputs[i]);
            }

            // Paso 5: Actualizar los pesos de la capa oculta
            for (int i = 0; i < nn.hiddenSize; i++)
            {
                for (int j = 0; j < nn.inputSize; j++)
                {
                    nn.hiddenWeights[i][j] -= nn.learningRate * hiddenErrors[i] * input[j];
                }
            }

            // Devuelve el error cuadrático medio
            return Math.Pow(output - target, 2);
        }

        private double SigmoidPrime(double x) => x * (1 - x);

    }
}
