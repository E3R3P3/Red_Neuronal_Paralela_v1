using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Red_Neuronal_Paralela_v1
{
    public class NeuralNetworkData
    {
        public double[][] HiddenWeights { get; set; }
        public double[] OutputWeights { get; set; }
        public int InputSize { get; set; }
        public int HiddenSize { get; set; }
    }
}
