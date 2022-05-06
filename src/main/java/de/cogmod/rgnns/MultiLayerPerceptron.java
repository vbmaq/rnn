package main.java.de.cogmod.rgnns;

import main.java.de.cogmod.rgnns.misc.LearningListener;
import main.java.de.cogmod.rgnns.misc.Tools;

import java.util.Random;

/**
 * Simple implementation of a multilayer perceptron with 
 * sigmoidal neurons and trainable biases.
 * 
 * @author Sebastian Otte
 */
public class MultiLayerPerceptron {
    
    public final static double BIAS = 1.0;
    
    private int layersnum;
    private int inputsize;
    private int weightsnum;
    private int[] layer;
    private double[][] net;
    private double[][] act;
    private double[][] bwbuffer;
    private double[][] delta;
    private double[][][] weights;
    private boolean[] usebias;
    private double[][][] dweights;
    
    private static int[] join(final int i1, final int i2, final int ...in) {
        final int[] result = new int[2 + in.length];
        //
        result[0] = i1;
        result[1] = i2;
        //
        for (int i = 0; i < in.length; i++) {
            result[i + 2] = in[i];
        }
        //
        return result;
    }
    
    public void setBias(final int layer, final boolean bias) {
        assert(layer > 0 && layer < this.layersnum);
        this.usebias[layer] = bias;
    }
    
    public boolean getBias(final int layer) {
        assert(layer > 0 && layer < this.layersnum);
        return this.usebias[layer];
    }
    
    /**
     * Constructor of the MLP class. The function signature ensures that there is
     * at least an input and an output layer (in this case layern.length would be 0). 
     * Otherwise layer2 would be the first hidden layer and layern would contain
     * at least the output layer and, optionally, further hidden layers. Effectively,
     * the firstly given layer defines the input layer size, the lastly given number
     * defines the output layer size, and the numbers in between define the hidden 
     * layers sizes, accordingly.
     * 
     * @param input Number of input neurons.
     * @param layer2 Layer of hidden/output neurons.
     * @param layern A list of further hidden and the output layer.
     */
    public MultiLayerPerceptron(
        final int input, final int layer2, final int... layern 
    ) {
        //
        this.inputsize = input;
        this.layer     = join(input, layer2, layern);
        this.layersnum = this.layer.length;
        //
        // set up buffers.
        //
        this.net      = new double[this.layersnum][];
        this.act      = new double[this.layersnum][];
        this.delta    = new double[this.layersnum][];
        this.bwbuffer = new double[this.layersnum][];
        this.usebias  = new boolean[this.layersnum];
        this.weights  = new double[this.layersnum][][];
        this.dweights = new double[this.layersnum][][];
        //
        this.weights[0]  = null;
        this.dweights[0] = null;
        //
        int sumweights = 0;
        //
        for (int l = 0; l < this.layersnum; l++) {
            //
            this.usebias[l] = true;
            //
            if (l > 0) {
                //
                // we don't need the net buffer for input neurons.
                //
                this.net[l]      = new double[this.layer[l]];
                this.delta[l]    = new double[this.layer[l]];
                this.bwbuffer[l] = new double[this.layer[l]];
            }
            this.act[l] = new double[this.layer[l]];
            //
            // The weights are arranged such that the first
            // index refers to the layer at which the corresponding 
            // "link" target to. For instance, this.weights[1] addresses
            // the weights from layer 0 to layer 1. Accordingly, 
            // this.weights[0] is undefined.
            //
            if (l > 0) {
                //
                // the plus + 1 used to model the bias weights.
                //
                this.weights[l]  = new double[this.layer[l - 1] + 1][this.layer[l]];
                this.dweights[l] = new double[this.layer[l - 1] + 1][this.layer[l]];
                //
                sumweights += (this.layer[l - 1] + 1) * (this.layer[l]);
            }
            //
        }
        //
        this.weightsnum = sumweights;
    }
    
    public double[][] getAct() { return this.act; }
    public double[][] getNet() { return this.net; }
    public double[][] getDelta() { return this.delta; }
    
    private static double sigmoid(final double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    private static double sigmoidDx(final double x) {
        final double sig = 1.0 / (1.0 + Math.exp(-x));
        return sig * (1.0 - sig);
    }

    /**
     * Computes the forward pass, i.e., propagates an input 
     * vector through the network to the output layer.
     * @param input Input vector.
     * @return Output vector.
     */
    public double[] forwardPass(final double[] input) {
        //
        // store input.
        //
        for (int i = 0; i < input.length; i++) {
            this.act[0][i] = input[i];
        }
        //
        // compute output layer-wise. start with the first
        // hidden layer (or the outputlayer if there is no
        // hidden layer).
        //
        for (int l = 1; l < this.layersnum; l++) {
            //
            // first compute all the net (integrate the inputs) values and activations.
            //
            final int layersize    = this.layer[l];
            final int prelayersize = this.layer[l - 1];
            //
            for (int j = 0; j < layersize; j++) {
                //
                // eventually initialize netj with the weighted bias.
                //
                double netj = 0;
                //
                if (this.usebias[l]) {
                    netj = BIAS * this.weights[l][prelayersize][j];
                }
                //
                for (int i = 0; i < prelayersize; i++) {
                    netj += this.act[l - 1][i] * this.weights[l][i][j];
                }
                //
                this.net[l][j] = netj;
            }
            //
            // now we compute the activations of the neurons in the current layer
            // (this could have also be done in the previous loop).
            //
            for (int j = 0; j < layersize; j++) {
                this.act[l][j] = sigmoid(this.net[l][j]);
            }            
        }
        //
        // return output layer activation as output.
        //
        return this.act[this.act.length - 1].clone();
    }
    
    public void backwardPass(final double[] target) {
        //
        // inject the output/target discrepancy into this.bwbuffer. Note that
        // this.bwbuffer functions analogously to this.net but is used to
        // store into "back-flowing" inputs (deltas).
        //

        //
        // back-propagate the error through the network -- we compute the deltas --
        // starting with the output layer.
        //

        // (VM) Plan: foreach neuron k of output layer, calculate the output deltas
        // (VM) I'm not sure if the instruction means to save all the deltas in this.bwbuffer or in this.delta :(
        // I'll just assume the deltas are saved in this.bwbuffer. Otherwise, just change this.bwbuffer to this.deltas
        // in the ff implementation...

        double[] prediction = this.act[layersnum - 1];
        int outputSize = this.layer[layersnum - 1];
        for(int k=0; k<outputSize; k++){
            this.delta[this.layer.length - 1][k] = sigmoidDx(this.net[this.net.length-1][k])
                    * (prediction[k] - target[k]);              // delta_k (eq. 25)
//            this.bwbuffer[layersnum - 1][k] = prediction[k] - target[k];

        }

        // calculate delta for the hidden layers
        // do it backwards: start with second last layer and end with first hidden layer
        for(int l= this.layersnum - 2; l>0; l--){               // iterate from last hidden layer to first hidden layer
            for(int j=0; j < this.layer[l]; j++){               // iterate through j neurons in current layer l
                double sum = 0;                                 // <-- second term of δ_h (eq. 26)
                for (int k=0; k < this.layer[l+1]; k++){        // iterate through k neurons in next layer l+1
                    sum += this.weights[l+1][j][k] * this.delta[l+1][k];
//                    bwbuffer[l][j] += weights[l+1][j][k] * bwbuffer[l+1][k];
                }
                this.delta[l][j] = sigmoidDx(this.net[l][j])* sum;  // delta_h (eq. 26)
            }
        }

        // 
        // Compute the weights derivatives.
        //
        // this.dweights !!!!
        // ...

        for(int l=1; l <= this.layersnum - 1; l++){          // first hidden layer to output
            for(int j=0; j< this.layer[l]; j++){            // neuron j=0... in current layer
                for(int i=0; i< this.layer[l-1]; i++){      // neuron i=0... in previous layer
                    var xi = this.act[l-1][i];       // activation of neuron i in previous layer
                    var deltaj = this.delta[l][j];  // delta of neuron j in current layer
                    this.dweights[l][i][j] = xi * deltaj;
                }
            }
        }



    }
    
    /**
     * Initializes the weights randomly and normal distributed with
     * std. dev. 0.1.
     * @param rnd Instance of Random.
     */
    public void initializeWeights(final Random rnd, final double stddev) {
        for (int l = 1; l < this.weights.length; l++) {
            for (int i = 0; i < this.weights[l].length; i++) {
                for (int j = 0; j < this.weights[l][i].length; j++) {
                    this.weights[l][i][j] = rnd.nextGaussian() * stddev;
                }
            }
        }
    }
    
    public int getWeightsNum() {
        return this.weightsnum;
    }
    
    public void writeWeights(final double[] weights) {
        int idx = 0;
        for (int l = 1; l < this.weights.length; l++) {
            for (int i = 0; i < this.weights[l].length; i++) {
                for (int j = 0; j < this.weights[l][i].length; j++) {
                    this.weights[l][i][j] = weights[idx++];
                }
            }
        }
    }

    public void readWeights(final double[] weights) {
        int idx = 0;
        for (int l = 1; l < this.weights.length; l++) {
            for (int i = 0; i < this.weights[l].length; i++) {
                for (int j = 0; j < this.weights[l][i].length; j++) {
                    weights[idx++] = this.weights[l][i][j]; 
                }
            }
        }
    }
    
    public void readDWeights(final double[] dweights) {
        int idx = 0;
        for (int l = 1; l < this.dweights.length; l++) {
            for (int i = 0; i < this.dweights[l].length; i++) {
                for (int j = 0; j < this.dweights[l][i].length; j++) {
                    dweights[idx++] = this.dweights[l][i][j]; 
                }
            }
        }
    }
    
    /**
     * Stochastic gradient descent.
     * @param rnd Instance of Random.
     * @param input Input vectors.
     * @param target Target vectors.
     * @param epochs Number of epochs.
     * @param learningrate Value for the learning rate.
     * @param momentumrate Value for the momentum rate.
     * @param listener Listener to observe the training progress.
     * @return The final epoch error.
     */
    public double trainStochastic(
        final Random rnd, 
        final double[][] input, 
        final double target[][],
        final double epochs,
        final double learningrate,
        final double momentumrate,
        final LearningListener listener
    ) {
        //
        final double[] weights           = new double[this.weightsnum];
        final double[] dweights          = new double[this.weightsnum];
        final double[] weightsupdate     = new double[this.weightsnum];
        //
        this.readWeights(weights);
        //
        // create initial index permutation.
        //
        final int[] indices = new int[input.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        //
        double error = Double.POSITIVE_INFINITY;
        //
        // epoch loop.
        //

        double[][][] prev_dweights;

        for (int i = 0; i < epochs; i++) {
            //
            // shuffle indices.
            //
            Tools.shuffle(indices, rnd);
            //
            double errorsum = 0.0;
            //
            // train all samples in online manner, i.e. iterate over all samples
            // while considering the shuffled order and update the weights 
            // immediately after each sample
            //

            for (int index: indices
                 ) {
                double[] pred = forwardPass(input[index]);

                errorsum += RMSE(pred, target[index]);

                prev_dweights = this.dweights.clone();

//                System.out.println("Previous weights:" + weights);
                backwardPass(target[index]);
//                System.out.println("Current weights:" + weights);

                for(int l=1; l<this.layersnum; l++){                // iterate from 1st hidden layer to output layer
                    for (int j=0; j<this.layer[l]; j++){            // j neurons of current layer
                        for (int ii=0; ii<this.layer[l-1]; ii++){   // ii neurons of previous layer
                            double momentumTerm = momentumrate * prev_dweights[l][ii][j];     // momentumRate * previous weight gradient
                            double dw = -learningrate * this.dweights[l][ii][j];
                            this.weights[l][ii][j] += dw;//+ momentumTerm;

                        }
                    }
                }

            }



            // ...
            
            //
            error = errorsum / (double)(input.length);
            if (listener != null) listener.afterEpoch(i + 1, error);
        }
        //
        return error;
    }
    
    
    /**
     * Computes the RMSE of the current output and
     * a given target vector.
     * @param target Target vector.
     * @return RMSE value.
     */
    public static double RMSE(final double[] output, final double[] target) {
        //
        double error = 0;
        //
        for (int i = 0; i < target.length; i++) {
            final double e = output[i] - target[i];
            error += (e * e);
        }
        //
        return Math.sqrt(error / (double)(target.length));
    }
    
    
}