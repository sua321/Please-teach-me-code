import java.util.List;
import java.util.Random;
import java.util.ArrayList;

class ForwardPass{ 

    List<List<Double>> weights_In_To_Hid; //Wights between inputlayer to hiddenlayer
    List<Double> weights_Hid_To_Out; //Weights between hiddenlayer to outputlayer
    List<Double> hidden_act; // For update hiddenlayer weights ,this "Hiddenlayer activation input " need
    double outBias; // Bias for outputlayer
    List<Double> hid_Bias; // Bias for hidenlayer neurons
    int no_Of_In;
    int no_Of_Hid_Ne;
public ForwardPass(int no_Of_In,int no_Of_Hid_Ne){ //no of input neurons and Hidden neurons
    weights_In_To_Hid = new ArrayList<>();
    weights_Hid_To_Out = new ArrayList<>();
    hidden_act = new ArrayList<>();
    hid_Bias = new ArrayList<>();

    Random random = new Random();
    if(no_Of_Hid_Ne < 1 || no_Of_In < 1)
        throw new IllegalArgumentException("The value of no of inputs and hidden neuron should be atleast 1");
    // Initalizing Veriables
    for(int i = 0; i < no_Of_Hid_Ne; i++){
        List<Double> row = new ArrayList<>(); // For adding rows in 2d array
        for(int j = 0; j< no_Of_In; j++)
        row.add(random.nextDouble()); // initalizing specific rows

        weights_In_To_Hid.add(row);   // Adding rows
        weights_Hid_To_Out.add(random.nextDouble());

        hid_Bias.add(random.nextDouble()/10);
    }
    this.outBias = random.nextDouble()/10;
    this.no_Of_Hid_Ne = no_Of_Hid_Ne;
    this.no_Of_In = no_Of_In;
}
    public double sigmoid(double inputs){
            return(1 / (1 + Math.exp(-inputs)));
    }

    public double prediction(List<Double> inputs){
        hidden_act.clear(); // Clear activations from previous call
        // forward pass for hiddenlayer

        double sumOut = outBias; // this first holds Bias of output neuron then holds weighted and biased output
        for(int i = 0; i < no_Of_Hid_Ne; i++){
            double sumHid = hid_Bias.get(i); // this first hold one hidden neuron's bias and then holds sum of all weights * in[puts of that neuron]
            for(int j = 0; j< no_Of_In; j++)
                sumHid += weights_In_To_Hid.get(i).get(j) * inputs.get(j);
            hidden_act.add(sigmoid(sumHid));
        }

        for(int i = 0; i < hidden_act.size();i++)
            sumOut += hidden_act.get(i) * weights_Hid_To_Out.get(i);
        return sigmoid(sumOut);
    }
}
 