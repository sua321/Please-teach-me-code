import java.util.List;
import java.util.ArrayList;

class Backpropagation{ 
    double learningRate;
    double outDelta; // Output neuron Delta
    List<Double> hid_Delta; // Hidden layer deltas
    int no_Of_Hid_Ne; // to set no of hidden neurons
    ForwardPass forward;
    public Backpropagation
          ( double learningRate, int no_Of_Hid_Ne){

                this.learningRate = learningRate;
                this.no_Of_Hid_Ne = no_Of_Hid_Ne;
                this.hid_Delta = new ArrayList<>();
            }

    public double derivative(double predicted){   // HiddenLayer argument to parameter each time

       
        return  (predicted * (1 - predicted));
         
    }

    public void training(int epoches, List<List<Double>> inputs, double[] output){ //Training Foward Pass
        int epoch;
         forward = new ForwardPass(inputs.get(0).size(),no_Of_Hid_Ne);
        double error ;
        for(epoch = 0; epoch < epoches;epoch++){
            double sumSquaredError = 0;
            for(int i = 0; i < inputs.size(); i++){
                List<Double> input = inputs.get(i);
                double predicted = forward.prediction(input);   // Prediction pack
                double derivated = derivative(predicted);
                
                    error=(output[i] - predicted);
                    sumSquaredError += error * error;
                    if( Math.abs(error) > 1e-9 ){ // Use a small tolerance instead of != 0 for floating point
                        outDelta = error * derivated;  // Output Neuron delta calculation
                        hid_Delta.clear();
                        for(int k = 0; k < forward.weights_Hid_To_Out.size(); k++) // loop runs until k < no. of hidden layer neurons
                        hid_Delta.add((outDelta * forward.weights_Hid_To_Out.get(k)) * (forward.hidden_act.get(k) *(1 - forward.hidden_act.get(k)))); // Hidenlayer neuron Delta
                    }
                    
                    if( Math.abs(error) > 1e-9 ){ 
                        for(int l = 0; l < forward.weights_Hid_To_Out.size(); l++){
                            forward.weights_Hid_To_Out.set(l, back_Hid_To_Out(outDelta, forward.hidden_act.get(l), forward.weights_Hid_To_Out.get(l))); // in this i am gonna set new weight for all the weight that in between hiddenlayer and outputlayer
                        }

                        // adding new weights to the weights and updating Bias that are in between input and hiddenlayers
                        for(int m = 0; m< forward.weights_In_To_Hid.size();  m++){
                            forward.weights_In_To_Hid.set(m, back_In_To_Hid(hid_Delta.get(m), input, forward.weights_In_To_Hid.get(m)));
                            //Bias
                            forward.hid_Bias.set(m, forward.hid_Bias.get(m) + (learningRate * hid_Delta.get(m)));
                        }    
                        forward.outBias += (learningRate * outDelta);

                    }
                        
            }
            if (inputs.size() > 0) {
                double mse = sumSquaredError / inputs.size();
                // Print MSE every N epochs (e.g., 1000) or on the last epoch
                if ( epoch % 1000 == 0 || epoch == epoches - 1) {
                     System.out.println("Epoch [" + epoch + "/" + (epoches-1) + "] - MSE: " + String.format("%.8f", mse) );
                }
            }
        }
    }

    // Backpropagation between hiddenlayer and outputlayer
    public double back_Hid_To_Out(double delta, double input, double weight){ 
        
        return weight + (learningRate * delta * input);
    } 

    // Back propabation between input and hidden layers
    public List<Double> back_In_To_Hid(double delta, List<Double> inputs, List<Double> weights){  
        List<Double> newWeights = new ArrayList<>();
        for(int i = 0; i< inputs.size(); i++){  // This calculation return changed weights of one hidden neuron 
           newWeights.add( weights.get(i) + (learningRate * delta * inputs.get(i)));
        }
        return newWeights;
    }

    // Input
    public void input() {
        List<List<Double>> inputs = new ArrayList<>();
double[][] inputsSet = {
    {0.1, 0.1}, // Diff = 0.0 -> Output = 0.0
    {0.9, 0.9}, // Diff = 0.0 -> Output = 0.0
    {0.5, 0.5}, // Diff = 0.0 -> Output = 0.0
    {0.1, 0.6}, // Diff = 0.5 -> Output = 4 * 0.5 * (1 - 0.5) = 1.0
    {0.8, 0.3}, // Diff = 0.5 -> Output = 4 * 0.5 * (1 - 0.5) = 1.0
    {0.2, 0.7}, // Diff = 0.5 -> Output = 4 * 0.5 * (1 - 0.5) = 1.0
    {0.1, 0.3}, // Diff = 0.2 -> Output = 4 * 0.2 * (1 - 0.2) = 0.8 * 0.8 = 0.64
    {0.7, 0.9}, // Diff = 0.2 -> Output = 4 * 0.2 * (1 - 0.2) = 0.64
    {0.6, 0.4}, // Diff = 0.2 -> Output = 4 * 0.2 * (1 - 0.2) = 0.64
    {0.9, 0.2}, // Diff = 0.7 -> Output = 4 * 0.7 * (1 - 0.7) = 2.8 * 0.3 = 0.84
    {0.1, 0.8}, // Diff = 0.7 -> Output = 4 * 0.7 * (1 - 0.7) = 0.84
    {0.4, 0.9}  // Diff = 0.5 -> Output = 4 * 0.5 * (1 - 0.5) = 1.0
};
        for(int i = 0; i< inputsSet.length; i++){
            List<Double> rows = new ArrayList<>(); // no of rows that in inputs

            for(int j = 0; j < inputsSet[i].length; j++)
                rows.add(inputsSet[i][j]);
            inputs.add(rows);
        }

        double[] output = {
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.64, 0.64, 0.64, 0.84, 0.84, 1.0
        }; // Training outputs

        // Training
        training(110000, inputs, output);
        // final testing
        double testingInputs[][] = {
            {0.2, 0.2}, // Diff = 0.0 -> Expected = 0.0
            {0.3, 0.8}, // Diff = 0.5 -> Expected = 1.0
            {0.9, 0.7}, // Diff = 0.2 -> Expected = 0.64
            {0.1, 0.9}, // Diff = 0.8 -> Expected = 4 * 0.8 * (1 - 0.8) = 3.2 * 0.2 = 0.64
            {0.6, 0.1}  // Diff = 0.5 -> Expected = 1.0
        };
        double testingOutputs[] = {
            0.0, 1.0, 0.64, 0.64, 1.0
        };

       for(int j = 0; j < testingInputs.length; j++){
        List<Double> rows = new ArrayList<>(); // no of rows that in inputs
        for(int k = 0; k< testingInputs[j].length; k++)
            rows.add(testingInputs[j][k]);
        double result = forward.prediction(rows);

        double epsilon = 1e-6; // A small tolerance value
        if (Math.abs(testingOutputs[j] - result) > epsilon)
            System.out.println("Output should be " + testingOutputs[j] + " insted of " + String.format("%.8f", result) );
        else 
            System.out.println("Congratulations the output " + String.format("%.8f", result) + " is correct!");
        
       }
       

        
    }    

    public static void main(String[] args) {
        Backpropagation bp = new Backpropagation(0.1, 8);
        bp.input();
    }

}