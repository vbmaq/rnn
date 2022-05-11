package main.java.de.cogmod.rgnns;

import main.java.de.cogmod.rgnns.misc.BasicLearningListener;

import java.util.Random;

/**
 * @author Sebastian Otte
 */
public class MLPXOR {
    
    public static void main(String[] args) {
        //
        final double[][] input = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        final double[][] target = {
            {0}, {1}, {1}, {0}
        };
        //
        final Random rnd = new Random(100L);
        //
        // set up network. biases are used by default, but
        // be deactivated using net.setBias(layer, false),
        // where layer gives the layer index (1 = the first hidden layer).
        //

        int hiddenL = 50; //==>0.0069
        double lr = 1;
        double mr = 1;

        hiddenL = 20; //==>0.0218       // without bias
        lr = 0.1;
        mr = 0.5;

        hiddenL = 30; //==>0.0272
        lr = 0.1;
        mr = 0.5;

        hiddenL = 40; //==>0.0275
        lr = 0.1;
        mr = 0.5;

        hiddenL = 30; //==>0.0145
        lr = 0.5;
        mr = 0.5;

        hiddenL = 30; //==>0.0107
        lr = 0.75;
        mr = 0.5;

        hiddenL = 30; //==>0.009    < 0.01
        lr = 1;
        mr = 0.5;

        hiddenL = 30; //==>0.0078
        lr = 1;
        mr = 1;

        hiddenL = 50; //==>0.0077       // with bias
        lr = 0.5;
        mr = 1;




        final MultiLayerPerceptron net = new MultiLayerPerceptron(2, hiddenL, 1);
        //
        // perform training.
        //
        final int epochs = 10000;         // don't change this value!
        final double learningrate = lr;
        final double momentumrate = mr;
        //
        // generate initial weights.
        //
        net.initializeWeights(rnd, 0.1);
        //
        net.trainStochastic(
            rnd, 
            input,
            target,
            epochs,
            learningrate,
            momentumrate,
            new BasicLearningListener()
        );
        //
    }

}
