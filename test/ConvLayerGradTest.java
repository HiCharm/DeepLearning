package test;

import source.Layer.ConvLayer;
import source.Tensor;
import java.util.function.Function;

public class ConvLayerGradTest {

    static void assertClose(double expected, double actual, double eps, String msg) {
        if (Double.isNaN(expected) || Double.isNaN(actual) || Math.abs(expected - actual) > eps) {
            throw new AssertionError(msg + " expected=" + expected + " actual=" + actual);
        }
    }

    static double sumOutput(ConvLayer cl) {
        double s = 0.0;
        for (int d = 0; d < cl.output.getDepth(); d++) {
            s += cl.output.get(d).sum();
        }
        return s;
    }

    public static void main(String[] args) {
        try {
            // prepare input a (3 x 5 x 5)
            Tensor a = new Tensor(3, 5, 5);
            double[][][] avals = new double[][][]{
                {{0,1,1,0,2},{2,2,2,2,1},{1,0,0,2,0},{0,1,1,0,0},{1,2,0,0,2}},
                {{1,0,2,2,0},{0,0,0,2,0},{1,2,1,2,1},{1,0,0,0,0},{1,2,1,1,1}},
                {{2,1,2,0,0},{1,0,0,1,0},{0,2,1,0,1},{0,1,2,2,2},{2,1,0,0,1}}
            };
            for (int d = 0; d < 3; d++)
                for (int i = 0; i < 5; i++)
                    for (int j = 0; j < 5; j++)
                        a.set(d, i, j, avals[d][i][j]);

            // create ConvLayer: input 5x5, channel 3, filter 3x3, 2 filters, stride1, zp=2
            Function<Double, Double> idF = x -> x;
            Function<Double, Double> idB = x -> 1.0;
            ConvLayer cl = new ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, 0.001, idF, idB);

            // set filters[0].weights
            double[][][] w0 = new double[][][]{
                {{-1,1,0},{0,1,0},{0,1,1}},
                {{-1,-1,0},{0,0,0},{0,-1,0}},
                {{0,0,-1},{0,1,0},{1,-1,-1}}
            };
            for (int d = 0; d < 3; d++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        cl.filters.get(0).weights.set(d, i, j, w0[d][i][j]);
            cl.filters.get(0).bias = 1;

            // set filters[1].weights
            double[][][] w1 = new double[][][]{
                {{1,1,-1},{-1,-1,1},{0,-1,1}},
                {{0,1,0},{-1,0,-1},{-1,1,0}},
                {{-1,0,0},{-1,0,1},{-1,0,0}}
            };
            for (int d = 0; d < 3; d++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        cl.filters.get(1).weights.set(d, i, j, w1[d][i][j]);

            // forward
            cl.forward(a);
            // set padded_input because bpGradient uses it
            cl.padded_input = cl.padding(a, cl.zp);

            // sensitivity array = ones same shape as output
            Tensor sens = new Tensor(cl.output.getDepth(), cl.output.getHeight(0), cl.output.getWidth(0));
            for (int d = 0; d < sens.getDepth(); d++)
                for (int i = 0; i < sens.getHeight(d); i++)
                    for (int j = 0; j < sens.getWidth(d); j++)
                        sens.set(d, i, j, 1.0);

            // initialize weights_grad holder
            cl.filters.get(0).weights_grad = new Tensor(cl.channel, cl.filterHeight, cl.filterWidth);
            cl.filters.get(1).weights_grad = new Tensor(cl.channel, cl.filterHeight, cl.filterWidth);

            // compute analytic gradients
            cl.bpGradient(sens);

            double eps = 1e-4;
            // numerical gradient check for filter 0
            for (int d = 0; d < cl.filters.get(0).weights.getDepth(); d++) {
                for (int i = 0; i < cl.filters.get(0).weights.getHeight(d); i++) {
                    for (int j = 0; j < cl.filters.get(0).weights.getWidth(d); j++) {
                        double orig = cl.filters.get(0).weights.get(d, i, j);
                        // f(w+eps)
                        cl.filters.get(0).weights.set(d, i, j, orig + eps);
                        cl.forward(a);
                        double err1 = sumOutput(cl);
                        // f(w-eps)
                        cl.filters.get(0).weights.set(d, i, j, orig - eps);
                        cl.forward(a);
                        double err2 = sumOutput(cl);
                        // restore
                        cl.filters.get(0).weights.set(d, i, j, orig);

                        double numeric = (err1 - err2) / (2 * eps);
                        double analytic = cl.filters.get(0).weights_grad.get(d).get(i, j);
                        System.out.printf("weights(%d,%d,%d): numeric=%f analytic=%f\n", d, i, j, numeric, analytic);
                        assertClose(numeric, analytic, 1e-6, "gradient mismatch at ("+d+","+i+","+j+")");
                    }
                }
            }

            System.out.println("ConvLayerGradTest PASS");
        } catch (Throwable t) {
            System.err.println("ConvLayerGradTest FAIL: " + t.getMessage());
            t.printStackTrace();
            System.exit(1);
        }
    }
}
