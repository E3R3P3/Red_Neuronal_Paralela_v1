using System;
using System.Collections.Generic;
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
        var path = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\.."));
        var file = Path.Combine(path, "training_dataset.csv");
        var data = loader.LoadData(file);                
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

        // Guardar la red neuronal entrenada
        var modelPath = Path.Combine(path, "trained_model.json");
        nn.SaveModel(modelPath);
        Console.WriteLine($"Modelo guardado en: {modelPath}");

        // Cargar la red neuronal entrenada
        var loadedModel = NeuralNetwork.LoadModel(modelPath);
        Console.WriteLine($"Modelo cargado desde: {modelPath}");
        // Realizar una predicción
        var testPath = Path.Combine(path, "testing_dataset.csv");
        var testData = loader.LoadData(testPath);
        var (testInputs, testTargets) = loader.PrepareData(testData);
        var tn = new NeuralNetwork(inputSize: testInputs[0].Length, hiddenSize: 5);
        Test test = new Test();
        var testTesult = test.TestModel(tn, testInputs, testTargets);

        Console.WriteLine($"Error de prueba: {testTesult}");
        Console.WriteLine("Entrenamiento y prueba completados.");
    }
}