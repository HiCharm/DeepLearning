package test;

import source.Layer.MaxPoolingLayer;
import source.Tensor;

public class maxPoolingTest {

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
        System.out.println(name + " (" + m.getHeight() + "x" + m.getWidth() + "): ");
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
        // single channel 4x4 input, pool 2x2 stride2 -> output 2x2
        MaxPoolingLayer layer = new MaxPoolingLayer(4, 4, 1, 2, 2, 2);

        Tensor input = new Tensor(1, 4, 4);
        double v = 1.0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                input.set(0, i, j, v++);
            }
        }

        System.out.println("--- Forward MaxPool Test ---");
        printTensor(input, "input");

        layer.forward(input);
        printTensor(layer.output, "output");

        double[][] expected = new double[][]{{6,8},{14,16}};
        assertMatrixEquals(expected, layer.output.get(0), 1e-6, "forward output mismatch");
    }

    static void testBackward() {
        MaxPoolingLayer layer = new MaxPoolingLayer(4, 4, 1, 2, 2, 2);

        Tensor input = new Tensor(1, 4, 4);
        double v = 1.0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                input.set(0, i, j, v++);
            }
        }

        layer.forward(input);
        // sensitivity (gradient) from next layer
        Tensor sens = new Tensor(1, layer.output.getHeight(0), layer.output.getWidth(0));
        sens.set(0, 0, 0, 1.0);
        sens.set(0, 0, 1, 2.0);
        sens.set(0, 1, 0, 3.0);
        sens.set(0, 1, 1, 4.0);

        System.out.println("--- Backward MaxPool Test ---");
        printTensor(sens, "sensitivity");

        layer.backward(sens);
        printTensor(layer.deltaArray, "deltaArray");

        double[][] expectedDelta = new double[][]{
            {0,0,0,0},
            {0,1,0,2},
            {0,0,0,0},
            {0,3,0,4}
        };

        assertMatrixEquals(expectedDelta, layer.deltaArray.get(0), 1e-6, "backward delta mismatch");
    }

    public static void main(String[] args) {
        try {
            testForward();
            testBackward();
            System.out.println("MaxPoolingTest PASS");
        } catch (Throwable t) {
            System.err.println("MaxPoolingTest FAIL: " + t.getMessage());
            t.printStackTrace();
            System.exit(1);
        }
    }
}
