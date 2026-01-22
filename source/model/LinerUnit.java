package source.model;

import source.unit.Matrix;
import source.util.AdvanceMatrixMath;
import source.util.CsvReader;
import source.util.DataSpliter;
import source.util.MatrixMath;

public class LinerUnit{
    static int splitRatio = 80;
    static String dataPath = "d:\\work\\2026\\DeepLearning\\Data\\RegressionData\\y=100x+1.csv";
    static double learningRate = 0.0001;
    static int maxEpoch = 50;
    static Matrix weights = null;
    static double bias = 0.0;

    public static double stepFunction(double x) {
        return x;
    }
    // features： n * x
    // weights: x * 1
    // y_pred: n * 1
    public static void forward(Matrix features, Matrix y_pred){
        AdvanceMatrixMath.multiply(features, weights, y_pred);
        MatrixMath.scalarAdd(bias, y_pred);
        for (int i = 0; i < y_pred.getHeight(); i++) {
            double activatedValue = stepFunction(y_pred.get(i, 0));
            y_pred.set(i, 0, activatedValue);
        }
    }

    // labls: n * 1
    // y_pred: n * 1
    // weights: x * 1
    // features: n * x
    // t_errors: 1 * n
    public static double backward(Matrix features, Matrix lables, Matrix y_pred){

        Matrix delta_weights = new Matrix(weights.getHeight(), 1);
        Matrix errors = new Matrix(1, lables.getHeight());

        // errors = lables - y_pred
        MatrixMath.subtract(lables, y_pred, errors);

        // t_errors = errors^T
        Matrix t_errors = MatrixMath.transpose(errors);

        // delta_weights = t_errors * features
        AdvanceMatrixMath.multiply(t_errors, features , delta_weights);

        // delta_weights = learningRate * delta_weights
        MatrixMath.multiply(learningRate, delta_weights);

        // delta_weights = delta_weights^T
        delta_weights = MatrixMath.transpose(delta_weights);

        // weights = weights + delta_weights
        MatrixMath.add(weights,delta_weights, true);

        // bias = bias + learningRate * sum(errors)
        bias += learningRate * MatrixMath.sum(errors, 0);

        return MatrixMath.sum(errors, 0);
    }

    public static void train(Matrix features, Matrix lables){
        Matrix y_pred = new Matrix(lables.getWidth(), lables.getHeight());
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            forward(features, y_pred);

            double error = backward(features, lables, y_pred);
            if (epoch % 1 == 0) {
                System.out.println("\nEpoch " + epoch + ", Error: " + error + ", \nWeight: " + weights.toString() + ", \nBias: " + bias);
            }
        }
    }

    public static void test(Matrix features, Matrix lables){
        Matrix y_pred = new Matrix(lables.getWidth(), lables.getHeight());
        forward(features, y_pred);
        int correctCount = 0;
        for (int i = 0; i < lables.getHeight(); i++) {
            if (y_pred.get(i, 0) - lables.get(i, 0) < 1) {
                correctCount++;
            }
        }
        double accuracy = (double) correctCount / lables.getHeight();
        System.out.println("Accuracy: " + accuracy);
    }

    public static void main(String[] args) {
        /**
         * 读取数据代码块
         * 分为trainFeatures, trainLabels, testFeatures, testLabels四个矩阵
         */
        Matrix data = CsvReader.readCsv(dataPath);
        Matrix trainData = new Matrix(data.getWidth(), data.getHeight() * splitRatio / 100);
        Matrix testData = new Matrix(data.getWidth(), data.getHeight() - trainData.getHeight());
        DataSpliter.splitData(data, trainData, testData);
        Matrix trainFeatures = new Matrix(data.getWidth() - 1, trainData.getHeight());
        Matrix trainLabels = new Matrix(1, trainData.getHeight());
        Matrix testFeatures = new Matrix(data.getWidth() - 1, testData.getHeight());
        Matrix testLabels = new Matrix(1, testData.getHeight());
        DataSpliter.featureLabelSplit(trainData, trainFeatures, trainLabels);
        DataSpliter.featureLabelSplit(testData, testFeatures, testLabels);

        // 模型部分
        // x * 1
        weights = new Matrix(1, trainFeatures.getWidth());
        MatrixMath.randomMatrix(-1.0, 1.0, weights);

        train(trainFeatures, trainLabels);
        test(testFeatures, testLabels);


    }
}
