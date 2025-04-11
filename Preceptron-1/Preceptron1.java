import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

class Preceptron1{
    List<Double> weights;
    double bias;
    double learningRate;
    int numberOfInputs;
    int epoches;
    public Preceptron1( int numberOfInputs, int epoches){
        if (numberOfInputs < 1) 
            throw new IllegalArgumentException("No of input must be atleast 1");
        
        this.numberOfInputs = numberOfInputs;
        this.weights = new ArrayList<Double>();
        this.learningRate = 0.01; 
        this.epoches = epoches;
        assigningRandomValues();
    }

    public void assigningRandomValues(){
        Random random = new Random();

        for(int i = 0; i < numberOfInputs;i++)
            weights.add(random.nextDouble());
        this.bias = random.nextDouble();
    }

    public double sigmoid(double wb){ // wb = weighted and biased inputs
        return 1/(1+Math.exp(-wb));
    }

    public double derivated(double p){ // p = predicted output 
        return p* (1- p);
    }

    // Calculating the output(Prediction)
    public double Prediction(double[] input){
        if(numberOfInputs != input.length)
            throw new IllegalArgumentException("Please enter the inputs that same size as you define before");
        double add = bias;
        for(int i= 0; i < input.length;i++)
            add += weights.get(i) * input[i];
        return sigmoid(add);
    }

    // Training and backward pass
    public void training(double[][] inputs,double[] output){
        int epoch;
        double mSE;   // Mean Squared Error (MSE) = The average of the squared errors. MSE = (1 / N) * Σ (predicted_i - target_i)² (Where N is inputs.length)
        double[] input;
        double delta;
        double sumSquaredError;

        // Training
        for(epoch = 0; epoch < epoches;epoch++){
            sumSquaredError = 0.0;
            for(int i = 0; i < inputs.length;i++){
                input = inputs[i];
                double predicted = Prediction(input);
                double error = predicted - output[i];
                delta = derivated(predicted) * error;
                sumSquaredError += error * error;
                // Corrected backpropagation: Subtracting the gradient component
                if(Math.abs(error) > 1e-9){ 
                    for(int j = 0; j < input.length; j++){
                        weights.set(j, weights.get(j) - input[j] * delta * learningRate) ; // Changed to subtraction
                    }
                    bias -= delta*learningRate; // Changed to subtraction
                }
            }
            mSE = sumSquaredError / inputs.length;
            // Print MSE every 1000 epochs or on the last epoch
            if ( epoch % 1000 == 0 || epoch == epoches - 1) {
                 System.out.println("Epoch [" + epoch +"]" + " MSE: " + mSE );
            }
        }
    }

    public static void main(String[] args) {
        int noOfInput = 2;
        double[][] inputs = { // Training inputs
            {0.3, 0.4},  
            {0.1, 0.9},  
            {0.8, 0.2},  
            {0.5, 0.5},  
            {0.9, 0.8}
        };
        double[] output = {0.1, 0.8, 0.3, 0.5, 0.9 }; // Training outputs

        Preceptron1 preceptron1 = new Preceptron1(noOfInput, 1000000);
        preceptron1.training(inputs, output); // training

        // final testing
       double testingInputs[][] ={{0.2, 0.1}, {0.6, 0.7}};
       double testingOutputs[] = {0.15,0.7};

       for(int i = 0; i < testingInputs.length; i++){
        double tolerance = 0.05;
        double result = preceptron1.Prediction(testingInputs[i]);
        System.out.println("Inputs: " + Arrays.toString(testingInputs[i]) + " Result: " +String.format("%.4f", result)
        + "CorrectOutput: " + testingOutputs[i]);

        if(Math.abs(result - testingOutputs[i]) > tolerance)
        System.out.println("stupid program. output should be " + testingOutputs[i] );
        else
        System.out.println( "Congratulations. you made it the output " + result +" is correct.");
       }
    }
}