package source.Layer;

import java.util.function.Function;

import source.Matrix;

public class FullyConnectLayer {
    public int inputSize;
    public int outputSize;
    public Function<Double, Double> activationFroward;
    public Function<Double, Double> activationBackward;
    Matrix weights;
    Matrix bias;
    Matrix weights_grad;
    Matrix bias_grad;
    Matrix input;
    public Matrix output;
    public Matrix delta;
    public FullyConnectLayer(int inputSize, int outputSize, Function<Double, Double> activationFroward, Function<Double, Double> activationBackward) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationFroward = activationFroward;
        this.activationBackward = activationBackward;
        this.weights = new Matrix(outputSize, inputSize);
        weights.randomize();
        this.bias = new Matrix(outputSize, 1);
        bias.fill(0);
    }
    public void forward(Matrix input) {
        this.input = input;
        this.output = weights.multiply(input);
        output.add(bias);
        for (int i = 0; i < output.getHeight(); i++) {
            for (int j = 0; j < output.getWidth(); j++) {
                output.set(i, j, activationFroward.apply(output.get(i, j)));
            }
        }
    }
    public void backward(Matrix delta) {
        weights_grad = delta.multiply(input.transpose());
        bias_grad = delta;
        this.delta = weights.transpose().multiply(delta);
        for (int i = 0; i < delta.getHeight(); i++) {
            for (int j = 0; j < delta.getWidth(); j++) {
                delta.set(i, j, delta.get(i, j) * activationBackward.apply(input.get(i, j)));
            }
        }
    }
    public void update(double learningRate) {
        weights_grad.multiply(learningRate);
        bias_grad.multiply(learningRate);
        weights.add(weights_grad);
        bias.add(bias_grad);
    }
}
