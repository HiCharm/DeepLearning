package util;

public class Sigmod {
    public static double forward(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Expect input x to be the activated sigmoid output y = sigmoid(z).
    // Then derivative sigmoid'(z) = y * (1 - y).
    public static double backward(double x) {
        double y = forward(x);
        return y* (1 - y);
    }
}
