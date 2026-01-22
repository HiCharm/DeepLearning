package source.util;

import source.unit.Matrix;

public class DataSpliter {
    public static void splitData(Matrix data,   Matrix trainData, Matrix testData) {
        for (int i = 0; i < trainData.getHeight(); i++) {
            for (int j = 0; j < data.getWidth(); j++) {
                trainData.set(i, j, data.get(i, j));
            }
        }
        for (int i = 0; i < testData.getHeight(); i++) {
            for (int j = 0; j < data.getWidth(); j++) {
                testData.set(i, j, data.get(trainData.getHeight() + i, j));
            }
        }
    }


    public static void featureLabelSplit(Matrix data, Matrix features, Matrix labels) {
        for (int i = 0; i < data.getHeight(); i++) {
            for (int j = 0; j < data.getWidth() - 1; j++) {
                features.set(i, j, data.get(i, j));
            }
            labels.set(i, 0, data.get(i, data.getWidth() - 1));
        }
    }
    
}
