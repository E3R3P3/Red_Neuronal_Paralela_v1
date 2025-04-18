using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Red_Neuronal_Paralela_v1;

class Program
{
    static void Main()
    {
        var loader = new DataLoader();
        var path = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..\\..\\..\\"));
        var file = Path.Combine(path, "training_dataset.csv");
        
        try
        {
            // Cargar y preparar datos
            var data = loader.LoadData(file);
            var (inputs, targets) = loader.PrepareData(data);

            // Crear e entrenar red neuronal
            var nn = new NeuralNetwork(inputSize: inputs[0].Length, hiddenSize: 5);
            var trainer = new Trainer();

            // Entrenamiento secuencial
            var watchSeq = Stopwatch.StartNew();
            trainer.Train(nn, inputs, targets, epochs: 100, parallel: false);
            watchSeq.Stop();

            // Entrenamiento paralelo
            var watchPar = Stopwatch.StartNew();
            trainer.Train(nn, inputs, targets, epochs: 100, parallel: true);
            watchPar.Stop();

            Console.WriteLine("\nResultados de entrenamiento:");
            Console.WriteLine($"Tiempo secuencial: {watchSeq.ElapsedMilliseconds}ms");
            Console.WriteLine($"Tiempo paralelo: {watchPar.ElapsedMilliseconds}ms");
            Console.WriteLine($"Mejora de rendimiento: {100 - (watchPar.ElapsedMilliseconds * 100 / watchSeq.ElapsedMilliseconds)}%\n");

            // Guardar modelo
            var modelPath = Path.Combine(path, "trained_model.json");
            nn.SaveModel(modelPath);
            Console.WriteLine($"Modelo guardado en: {modelPath}");

            // Cargar modelo y probar
            var loadedModel = NeuralNetwork.LoadModel(modelPath);
            Console.WriteLine($"Modelo cargado desde: {modelPath}");

            // Probar con datos de test
            var testPath = Path.Combine(path, "testing_dataset.csv");
            var testData = loader.LoadData(testPath);
            var (testInputs, testTargets) = loader.PrepareData(testData);

            var tester = new Test();
            var (avgError, predictions) = tester.TestModel(loadedModel, testInputs, testTargets, exportToCsv: true);

            // Mostrar resultados
            Console.WriteLine("\nPrimeras 5 predicciones:");
            Console.WriteLine("Entrada\t\t\tPredicción\tReal\tError");
            // for (int i = 0; i < 5 && i < predictions.Count; i++)
            // {
            //     var pred = predictions[i];
            //     Console.WriteLine($"[{string.Join(",", pred.InputFeatures.Select(f => f.ToString("F2"))}]\t" +
            //                       $"{pred.PredictedValue.ToString("F2")}\t" +
            //                       $"{pred.ActualValue.ToString("F2")}\t" +
            //                       $"{pred.Error.ToString("F4")}");
            // }

            Console.WriteLine($"\nError promedio en prueba: {avgError.ToString("F4")}");
            Console.WriteLine($"Predicciones exportadas en: {Path.Combine(path, "predictions.csv")}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine($"Detalles: {ex.StackTrace}");
        }
        
        Console.WriteLine("\nPresione cualquier tecla para salir...");
        Console.ReadKey();
    }
}