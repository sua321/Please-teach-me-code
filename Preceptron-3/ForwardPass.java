import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class ForwardPass{

    List<List<List<Double>>> weights;
    List<List<Double>> biases; // all layers biases
    Random random;
    int numLayers; // no of Layers
    List<List<Double>> preActivatedSums;  // holds weighted sum before neuron activation
    public ForwardPass(int numLayers){
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        random = new Random();
        preActivatedSums = new ArrayList<>();
        this.numLayers = numLayers;

    }

    public void networkInitializer(List<Integer> neuronSet,int input_size){ 
        List<List<Double>> layer; // holds the weights of the every neurons in the layer
        List<Double> row;  // holds every neuron's weights
        List<Double> bias; // single layer Biases
        for(int i = 0; i < numLayers; i++){   // adding the layer , i = layer number
            layer = new ArrayList<>();
            bias = new ArrayList<>();
            weights.add(layer);
            biases.add(bias);
            for(int j = 0; j < neuronSet.size(); j++){ // calculates weights and bias for single neuron , j = neuron number
                row = new ArrayList<>();
                layer.add(row);
                for(int k = 0; k < input_size; k++){    // k = weight's number (this value is atomic {atomic means every single weight is in the neuron so "k" represents it })
                    row.add(random.nextDouble());
                }
                if(i>0)
                input_size = neuronSet.get(i-1);
                bias.add(random.nextDouble()/10);
            }
            
        }
    }



        public double _ReLU(double input){
            return Math.max(0, input); // returns the maximum number

        }


        public List<List<Double>> layerForwardPass(List<Double> inputs){   // this method will return all hiddenlayer inputs And final output

            List<List<Double>> layerInputs = new ArrayList<>();  // at end of the the every single layer's loop(i) this store's neuron's output for back propagation {also stores the outputs at thr end of the list}
            double sums;  // first store's the "i" layer's "j" neuron's bias the stores the weighted and biased value of the input

            for(int i = 0; i < weights.size(); i++){             // layer loop , i = layer number

                List<Double> preSums = new ArrayList<>(); //holds weighted sums for an layer 
                List<Double> neuronOutputs = new ArrayList<>(); // this veriable stores the output of the neurons in the layer (after activation)
                for(int j = 0; j < weights.get(i).size();j++){    // neuron loop , j = neuron number
                    sums = biases.get(i).get(j);                // assigning the bias of neuron "j"
                    for(int k = 0; k < weights.get(i).get(j).size(); k++){   // k = weight's number (this value is atomic)
                        sums += weights.get(i).get(j).get(k) * inputs.get(k);
                    }
                    preSums.add(sums);
                    neuronOutputs.add(_ReLU(sums));

                }
                 layerInputs.add(neuronOutputs);
                 preActivatedSums.add(preSums);
                 inputs=neuronOutputs;
            }
            return layerInputs;

               


        }
}