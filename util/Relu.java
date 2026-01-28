package util;

public class Relu {
    public static double forward(double x) {
        return Math.max(0, x);
    }

    public static double backward(double x) {
        return x > 0 ? 1 : 0;
    }
}
