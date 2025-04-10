import java.util.Arrays;

class Preceptron{
    double[]weights;
    double bias;
    double learningRate;
    int epochs;
public Preceptron( double learningRate, int epochs){
    this.weights = new double[]{0.3,0.4};
    this.bias = 0.3;
    this.learningRate = learningRate;
    this.epochs = epochs;
}
// Sigmoid activation function(Method)
public double Sigmoid(double p){    //p = predicted 
    return 1 /(1+ Math.exp(-p)) ;
}

// Sigmoid derivative function for Back propergation(This tell neuron's confident level 0 - 0.25)
public double derivative(double p){
    return p * (1 - p);
}

// Weighted and Biased output with neuron activation
public double prediction(double[] input){
    double sum = bias;
    for(int i =0; i<2; i++)
    sum += weights[i] * input[i];
    double predicted = Sigmoid(sum);
    return predicted ;
}

// Back propagation with training
public void training(double[][] inputs, double[] outputs){
    double error = 0;
    int epoch;
    double confidence = 0;
    double predicted = 0;
    double errorTolerance = 0.00001;
    // Training
    for(epoch = 0; epoch < epochs;epoch++){
        
            for(int i = 0; i < inputs.length; i++){
               double[] input = inputs[i];
                predicted = prediction(input);
                error = outputs[i] - predicted;
                if (Math.abs(error) > errorTolerance){ // (error != 0)
                    confidence = derivative(predicted);
                    double delta = confidence * error;
                    for(int j= 0; j < weights.length; j++){
                        weights[j] += input[j] *delta * learningRate;
                    }
                    bias += delta* learningRate;
                }
            }
        System.out.println("Epoch: [" + epoch +"]" +" Error rate: "+ error + " Confidence: " + confidence * 400 +"%" );
        }
    }
public static void main(String[] args) {
    Preceptron neuron = new Preceptron(0.1, 10000);
    // Training inputs, outputs
    double[][] inputs = {{0.3,0.4},{0.1,0.2},{0.5,0.5},{0.8,0.1}};
    double[] output = {0.5,0.1,0.8,0.7};
    neuron.training(inputs, output);

    // Testing the neuron
    double[][] inputForTest = {{0.2,0.7},{0.6,0.5}};
    double[] outputForTest = {0.7,0.9};
    double testTolerance = 0.1;
    for(int i = 0; i < inputForTest.length; i++){
        double result = neuron.prediction(inputForTest[i]);
        if (Math.abs(outputForTest[i] - result) > testTolerance)  // this is my original statement but gemini asistent suggest to chang this ((outputForTest[i] - result) != 0)
            System.out.println("Inputs are: "+ Arrays.toString(inputForTest[i])+"\n"+"Stupid mechine the output  should be: " +outputForTest[i] + "not "+ result);
        else
        System.out.println("Inputs are: " + Arrays.toString(inputForTest[i]) + "\n"+  "Congratulations You made it. the output " + result + " is correct" );
    }
    


}
}
