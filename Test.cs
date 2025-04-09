using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Red_Neuronal_Paralela_v1
{
    public class Test
    {
        public double TestModel(NeuralNetwork nn, double[][] testInputs, double[] testTargets)
        {
            double totalError = 0;

            // Itera sobre los datos de prueba
            for (int i = 0; i < testInputs.Length; i++)
            {
                // Obtener la predicción para cada entrada
                double prediction = nn.Forward(testInputs[i]);

                // Calcular el error cuadrático medio (MSE)
                double error = Math.Pow(prediction - testTargets[i], 2);
                totalError += error;
            }

            // Calcular el error promedio (MSE promedio)
            return totalError / testInputs.Length;
        }
    }
}
