package test;

import source.Layer.ConvLayer;
import source.Tensor;
import util.Relu;

public class convLayerTest {

    static void assertEquals(double expected, double actual, double eps, String msg) {
        if (Double.isNaN(expected) || Double.isNaN(actual) || Math.abs(expected - actual) > eps) {
            throw new AssertionError(msg + " expected=" + expected + " actual=" + actual);
        }
    }

    static void assertMatrixEquals(double[][] expected, source.Matrix actual, double eps, String msg) {
        int h = actual.getHeight();
        int w = actual.getWidth();
        if (expected.length != h || expected[0].length != w) {
            throw new AssertionError(msg + " size mismatch");
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                assertEquals(expected[i][j], actual.get(i, j), eps, msg + " at ("+i+","+j+")");
            }
        }
    }

    static void printMatrix(source.Matrix m, String name) {
        System.out.println(name + " (" + m.getHeight() + "x" + m.getWidth() + "):");
        for (int i = 0; i < m.getHeight(); i++) {
            for (int j = 0; j < m.getWidth(); j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    static void printTensor(Tensor t, String name) {
        System.out.println(name + " depth=" + t.getDepth());
        for (int d = 0; d < t.getDepth(); d++) {
            System.out.println("-- depth " + d + " --");
            printMatrix(t.get(d), name + "[" + d + "]");
        }
    }

    static void testForward() {
        // input 1 channel 3x3
        ConvLayer layer = new ConvLayer(3, 3, 1, 2, 2, 1, 1, 0, 0.1, Relu::forward, Relu::backward);

        // set filter weights to [[1,0],[0,1]] and bias 0
        ConvLayer.Filter filter = layer.filters.get(0);
        filter.weights.set(0, 0, 0, 1);
        filter.weights.set(0, 0, 1, 0);
        filter.weights.set(0, 1, 0, 0);
        filter.weights.set(0, 1, 1, 1);
        filter.bias = 0;

        // input matrix 1..9
        Tensor input = new Tensor(1, 3, 3);
        double v = 1.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                input.set(0, i, j, v);
                v += 1.0;
            }
        }

        // visualization: show input and filter before forward
        System.out.println("--- Forward pass visualization ---");
        printTensor(input, "input");
        printTensor(filter.weights, "filter.weights (before)");
        System.out.println("filter.bias=" + filter.bias);

        layer.forward(input);

        // show output
        printTensor(layer.output, "output (after activation)");

        // expected convolution (before ReLU): [[6,8],[12,14]] -> after ReLU same
        double[][] expected = new double[][]{{6,8},{12,14}};

        assertMatrixEquals(expected, layer.output.get(0), 1e-6, "forward output mismatch");
    }

    static void testBackwardAndGradient() {
        ConvLayer layer = new ConvLayer(3, 3, 1, 2, 2, 1, 1, 0, 0.1, Relu::forward, Relu::backward);

        // same filter
        ConvLayer.Filter filter = layer.filters.get(0);
        filter.weights.set(0, 0, 0, 1);
        filter.weights.set(0, 0, 1, 0);
        filter.weights.set(0, 1, 0, 0);
        filter.weights.set(0, 1, 1, 1);
        filter.bias = 0;

        // input
        Tensor input = new Tensor(1, 3, 3);
        double v = 1.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                input.set(0, i, j, v);
                v += 1.0;
            }
        }

        // run forward to set input reference used by activationBackward
        layer.forward(input);
        // set padded_input (forward doesn't keep it)
        layer.padded_input = layer.padding(input, layer.zp);

        System.out.println("--- Backward pass visualization ---");
        printTensor(input, "input");
        printTensor(layer.padded_input, "padded_input");
        printTensor(filter.weights, "filter.weights (before)");

        // delta from next layer: ones 1x2x2
        Tensor delta = new Tensor(1, 2, 2);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                delta.set(0, i, j, 1.0);

        printTensor(delta, "incoming delta");

        // initialize weights_grad holder
        filter.weights_grad = new Tensor(layer.channel, layer.filterHeight, layer.filterWidth);

        // visualize expanded delta and flipped weights used internally
        source.Tensor expanded = layer.expandDeltaTensor(delta);
        printTensor(expanded, "expanded delta (stride applied)");
        source.Tensor flipped = layer.flipWeights180(filter);
        printTensor(flipped, "flipped weights (180)");

        // compute delta map (w.r.t input)
        layer.bpDeltaMap(delta);

        // expected input delta computed manually
        double[][] expectedDelta = new double[][]{
            {2,1,0},
            {1,1,0},
            {0,0,0}
        };

        System.out.println("delta w.r.t input:");
        printTensor(layer.delta, "layer.delta");

        assertMatrixEquals(expectedDelta, layer.delta.get(0), 1e-6, "input delta mismatch");

        // compute gradients
        layer.bpGradient(delta);

        // show gradients
        System.out.println("filter.weights_grad:");
        printTensor(filter.weights_grad, "weights_grad");
        System.out.println("filter.bias_grad=" + filter.bias_grad);

        // expected weights_grad [[12,16],[24,28]] and bias_grad 4
        double[][] expectedWg = new double[][]{{12,16},{24,28}};
        assertMatrixEquals(expectedWg, filter.weights_grad.get(0), 1e-6, "weights_grad mismatch");
        assertEquals(4.0, filter.bias_grad, 1e-6, "bias_grad mismatch");
    }

    public static void main(String[] args) {
        try {
            testForward();
            testBackwardAndGradient();
            System.out.println("ConvLayerTest PASS");
        } catch (Throwable t) {
            System.err.println("ConvLayerTest FAIL: " + t.getMessage());
            t.printStackTrace();
            System.exit(1);
        }
    }
}
