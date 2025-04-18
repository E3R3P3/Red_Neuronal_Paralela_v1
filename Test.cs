using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace Red_Neuronal_Paralela_v1
{
    public class Test
    {
        public (double, List<PredictionResult>) TestModel(NeuralNetwork nn, double[][] testInputs, double[] testTargets, bool exportToCsv = false)
        {
            double totalError = 0;
            var results = new List<PredictionResult>();
            var culture = CultureInfo.InvariantCulture;

            for (int i = 0; i < testInputs.Length; i++)
            {
                double prediction = nn.Forward(testInputs[i]);
                double realValue = testTargets[i];
                double error = Math.Pow(prediction - realValue, 2);
                totalError += error;

                results.Add(new PredictionResult(
                    InputFeatures: testInputs[i],
                    PredictedValue: prediction,
                    ActualValue: realValue,
                    Error: error
                ));
            }

            if (exportToCsv)
            {
                ExportPredictionsToCsv(results, "predictions.csv");
            }

            return (totalError / testInputs.Length, results);
        }

        private void ExportPredictionsToCsv(List<PredictionResult> results, string filePath)
        {
            var csvContent = new StringBuilder();
            
            // Encabezados
            csvContent.AppendLine("InputFeatures,PredictedValue,ActualValue,Error");
            
            // Datos
            foreach (var result in results)
            {
                string features = string.Join("|", result.InputFeatures.Select(f => f.ToString("F4")));
                csvContent.AppendLine(
                    $"{features}," +
                    $"{result.PredictedValue.ToString("F4")}," +
                    $"{result.ActualValue.ToString("F4")}," +
                    $"{result.Error.ToString("F4")}");
            }
            
            File.WriteAllText(filePath, csvContent.ToString());
        }
    }

    public record PredictionResult(
        double[] InputFeatures,
        double PredictedValue,
        double ActualValue,
        double Error
    );
}