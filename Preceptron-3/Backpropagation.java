import java.util.ArrayList;
import java.util.List;

public class Backpropagation {
    double learningRate;
    ForwardPass forward;
    List<Integer> neuronSet; // holds the number of neurons in each layer
    public Backpropagation(int layers,List<Integer> neuronSet, double learningRate){
        this.neuronSet = neuronSet;
        this.forward = new ForwardPass(layers);
        this.learningRate = learningRate;
    }

    public double derivetive(double input){   // Derivetive of ReLU
        return (input > 0) ? 1.0 : 0.0;

    }

    public List<List<Double>> allDelta(List<Double> outDeltas,List<List<List<Double>>> allWeights, List<List<Double>> preActivationOutputs){    // Note: in setting deltas for layers ,think The forward pass Output layer work as input layer in this
        List<List<Double>> deltas = new ArrayList<>();
        for(int i = 0; i < forward.numLayers-1; i++){
            deltas.add(new ArrayList<>());
        }
        deltas.add(outDeltas);
        for(int i = allWeights.size() - 1; i >= 0; i--){ // layers goes final to first(weight layers) 
           

                List<Double> hidLayerDeltas = new ArrayList<>();
                for(int j = 0; j < allWeights.get(i).get(0).size(); j ++){ // the weight's loop ( in (j) iam getting the weights of the first neuron(0) in the current(i) layer) )
                double sumDW = 0;
                    for(int k = 0; k < allWeights.get(i).size(); k++){ // neuron loop 
                       sumDW += deltas.get(i).get(k)* allWeights.get(i).get(k).get(j);
                    }
                    hidLayerDeltas.add(sumDW * derivetive(preActivationOutputs.get(i).get(j)));
    
                }
                if(i> 0)
                deltas.set(i-1, hidLayerDeltas);
            

        }

        return deltas;
    }

    public void backPropagation(List<List<Double>> inputsForEveryLayer,List<List<Double>> deltas){ // Calculate Gradients (How Much to Change Weights and Biases)
        for(int i = 0 ; i < forward.weights.size(); i++ ){ // layer loop
            for(int j = 0; j< forward.weights.get(i).size(); j++){ // neuron loop
                for(int k = 0; k < forward.weights.get(i).get(j).size(); k++){ // weights loop for each neuron
                    // weights update
                    forward.weights.get(i).get(j).set(k, forward.weights.get(i).get(j).get(k) - deltas.get(i).get(j)* inputsForEveryLayer.get(i).get(k)* learningRate);
                }
                // Bias Update
                forward.biases.get(i).set(j, forward.biases.get(i).get(j) - deltas.get(i).get(j) * learningRate);
            }

        }

    }

    public void Training (List<List<Double>> startingInput, List<List<Double>> realOutputs,int epoches){
        int epoch;
        List<Double> outputDelta;
        List<List<Double>> deltas;
        List<Double> preSum;
        forward.networkInitializer(neuronSet,startingInput.get(0).size());
        for(epoch = 0; epoch < epoches;epoch++){  // every epoch
            for(int i = 0; i < startingInput.size();i++){  // every set of inputs
                List<Double> inputs = startingInput.get(i);
                List<List<Double>> results = forward.layerForwardPass(inputs);  // results stores the outputs of every layer the last list is final output
                List<Double> prediction = new ArrayList<>();
                List<Double> output = new ArrayList<>(); // single set of real outputs for the single set of inputs
                prediction.addAll(results.get(results.size()-1));
                output.addAll(realOutputs.get(i));

                // Error and Output Layer Delta Calculation
                outputDelta = new ArrayList<>();
                preSum = forward.preActivatedSums.get(forward.preActivatedSums.size()-1);
                for(int j = 0; j < output.size(); j++){
                    double error = output.get(j) - prediction.get(j); 
                    double outDelta = error * derivetive(preSum.get(j));
                    outputDelta.add(outDelta);
                }
                // All Delta Calculation
                List<List<Double>> preActivationInForward = new ArrayList<>();
                preActivationInForward.addAll(forward.preActivatedSums);
                preActivationInForward.remove(preActivationInForward.size()-1);
                preActivationInForward.add(0,new ArrayList<>());
                List<List<Double>> allDeltas = allDelta(outputDelta,forward.weights ,preActivationInForward);

                // Weights and biases update
                results.add(0,inputs);   // 1
                results.remove(results.size()-1); // 2 , these one and to steps make the results(or inputs) more cleaner by adding first input and removing the last outputs
                backPropagation(results, allDeltas);

            }
            
            
        }
    }
}
