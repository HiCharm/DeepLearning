package util;

public class Sigmod {
    public static double forward(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double backward(double x) {
        double sig = forward(x);
        return sig * (1 - sig);
    }
}
