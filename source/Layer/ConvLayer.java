package source.Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import source.Matrix;
import source.Tensor;

public class ConvLayer {
    public int inputHeight;
    public int inputWidth;
    public int outputHeight;
    public int outputWidth;
    public int channel;
    public int filterHeight;
    public int filterWidth;
    public int filterNumber;
    public int stride;
    public int zp;
    public Function<Double, Double> activationFroward;
    public Function<Double, Double> activationBackward;

    public Tensor input;
    public Tensor output;
    public Tensor padded_input;
    public Tensor delta;

    public double learningRate;

    public class Filter {
        public Tensor weights;
        public double bias;
        public Tensor weights_grad;
        public double bias_grad;
        public Filter(int filterHeight, int filterWidth, int channel) {
            this.weights = new Tensor(channel, filterHeight, filterWidth);
            this.weights.randomize();
            this.bias = 0;
        }
        public void update(double learningRate) {
            weights_grad.multiply(learningRate);
            weights.add(weights_grad);
            bias += bias_grad * learningRate;
        }
    }

    public List<Filter> filters;

    public int calc_output_size(int inputSize, int filterSize, int zp, int stride) {
        return (inputSize - filterSize + 2*zp) / stride + 1;
    }

    public void conv(Matrix input, Matrix kernel, Matrix output, int stride, double bias) {
        for(int i = 0; i < output.getHeight(); i++) {
            for(int j = 0; j < output.getWidth(); j++) {
                double sum = 0;
                for(int k = 0; k < kernel.getHeight(); k++) {
                    for(int l = 0; l < kernel.getWidth(); l++) {
                            int x = i * stride + k;
                            int y = j * stride + l;
                            if(x >= 0 && x < input.getHeight() && y >= 0 && y < input.getWidth()) 
                                sum += input.get(x, y) * kernel.get(k, l);
                        }
                    }
                sum += bias;
                output.set(i, j, sum);
            }
        }
    }

    public Tensor padding(Tensor input, int zp) {
        if(zp == 0) {
            return input;
        }
        else {
            Tensor padded = new Tensor(input.getDepth(), input.getHeight(0) + 2*zp, input.getWidth(0) + 2*zp);
            for(int i = 0; i < input.getDepth(); i++) {
                for(int j = 0; j < input.getHeight(i); j++) {
                    for(int k = 0; k < input.getWidth(i); k++) {
                        padded.set(i, j+zp, k+zp, input.get(i, j, k));
                    }
                }
            }
            return padded;
        }
    }

    public ConvLayer(int inputHeight, int inputWidth, int channel, int filterHeight, int filterWidth, int filterNumber, int stride, int zp, double learningRate,Function<Double, Double> activationFroward, Function<Double, Double> activationBackward) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.channel = channel;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        this.filterNumber = filterNumber;
        this.filters = new ArrayList<>(filterNumber);
        for(int i = 0; i < filterNumber; i++) {
            filters.add(new Filter(filterHeight, filterWidth, channel));
        }
        this.stride = stride;
        this.zp = zp;
        this.activationFroward = activationFroward;
        this.activationBackward = activationBackward;
        this.outputHeight = calc_output_size(inputHeight, filterHeight, zp, stride);
        this.outputWidth = calc_output_size(inputWidth, filterWidth, zp, stride);
        this.output = new Tensor(filterNumber, outputHeight, outputWidth);
        this.learningRate = learningRate;
    }

    public void forward(Tensor input) {
        this.input = input;
        Tensor paddedTensor = padding(input, zp);

        for(int i = 0; i < filterNumber; i++) {
            Matrix output = new Matrix(outputHeight, outputWidth);
            Filter filter = filters.get(i);
            for(int j = 0; j < channel; j++) {
                Matrix inputChannel = paddedTensor.get(j);
                Matrix kernel = filter.weights.get(j);
                conv(inputChannel, kernel, output, stride, filter.bias);
                this.output.get(i).add(output);
            }
            this.output.get(i).apply(activationFroward);
        }
    }

    public Tensor expandDeltaTensor(Tensor delta) {
        Tensor expanded = new Tensor(delta.getDepth(), inputHeight - filterHeight + 1 + 2*zp, inputWidth - filterWidth + 1 + 2*zp);
        for(int i = 0; i < delta.getDepth(); i++) {
            for(int j = 0; j < delta.getHeight(i); j++) {
                for(int k = 0; k < delta.getWidth(i); k++) {
                    expanded.set(i, j*stride, k*stride, delta.get(i, j, k));
                }
            }
        }
        return expanded;
    }

    public Tensor flipWeights180(Filter filter) {
        Tensor flipped = new Tensor(filter.weights.getDepth(), filter.weights.getHeight(0), filter.weights.getWidth(0));
        for(int i = 0; i < filter.weights.getDepth(); i++) {
            for(int j = 0; j < filter.weights.getHeight(i); j++){
                for(int k = 0; k < filter.weights.getWidth(i); k++) {
                    flipped.set(i, filter.weights.getHeight(i)-1-j, filter.weights.getWidth(i)-1-k, filter.weights.get(i, j, k));
                }
            }
        }
        return flipped;
    }

    public void bpDeltaMap(Tensor delta) {
        Tensor expandedDelta = expandDeltaTensor(delta);
        Tensor paddedTensor = padding(expandedDelta, zp);
        this.delta = new Tensor(channel, inputHeight, inputWidth);

        for(int i = 0; i < filterNumber; i++) {
            Filter filter = filters.get(i);
            Tensor filpTensor = flipWeights180(filter);
            Tensor delta_i = new Tensor(channel, inputHeight, inputWidth);
            for(int j = 0; j < channel; j++) {
                conv(paddedTensor.get(j), filpTensor.get(j), delta_i.get(j), stride, 0);
            }
            this.delta.add(delta_i);
        }
        for(int i = 0; i < channel; i++){
            for(int j = 0; j < inputHeight; j++) {
                for(int kl = 0; kl < inputWidth; kl++){
                    this.delta.set(i, j, kl, this.delta.get(i, j, kl) * activationBackward.apply(this.input.get(i, j, kl)));
                }
            }
        }
    }
}
