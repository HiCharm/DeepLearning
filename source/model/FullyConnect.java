package source.model;

import java.util.List;

import Data.MINST.MNISTDataset.BatchData;

import java.util.ArrayList;

import source.Layer.FullyConnectLayer;
import util.*;
import source.Matrix;


public class FullyConnect {

    static List<FullyConnectLayer> layers;

    public static Matrix predict(Matrix input) {
        Matrix output = input;
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).forward(output);
            output = layers.get(i).output;
        }
        return output;
    }

    public static void update_weights(double learningRate) {
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).update(learningRate);
        }
    }

    public static void calc_gradient(Matrix labels) {
        FullyConnectLayer outputLayer = layers.get(layers.size() - 1);
        // 计算输出层的delta
        Matrix delta = new Matrix(outputLayer.outputSize, 1);
        for(int i = 0; i < delta.getHeight(); i++){
            delta.set(i,0, (labels.get(i,0) - outputLayer.output.get(i,0)) * outputLayer.activationBackward.apply(outputLayer.output.get(i,0)));
        }

        for(int i = layers.size() - 1; i >= 0; i--) {
            layers.get(i).backward(delta);
            delta = layers.get(i).delta;
        }
    }
    public static boolean train_one_sample(Matrix input, int label, double learningRate) {
        Matrix predict_label = predict(input);
        Matrix lables = new Matrix(10, 1);
        lables.set(label, 0, 1.0);
        calc_gradient(lables);
        update_weights(learningRate);
        double max_predict = 0;
        int max_label = 0;
        for(int i = 0; i < predict_label.getHeight(); i++){
            if(predict_label.get(i,0) > max_predict) {
                max_predict = predict_label.get(i,0);
                max_label = i;
            }
        }
        if(max_label == label) return true;
        return false;
    }

    public static void train(Matrix input, Matrix lables, double learningRate, int epoch) {
        for (int i = 0; i < epoch; i++) {
            System.out.println("epoch: " + i);
            int correctCount = 0;
            for(int j = 0; j < input.getHeight(); j++){
                if(train_one_sample(input.getRowAndTranspose(j), (int)lables.get(j,0), learningRate)){
                    correctCount++;
                }
                if(j % 100 == 0){
                    System.out.println("\ntrained sample count: " + j);
                    System.out.println("current accuracy: " + (double)correctCount / (j + 1) );
                }
            }
        }
    }

    public static void test(Matrix input, Matrix lables) {
        int correctCount = 0;
        for(int i = 0; i < input.getHeight(); i++){
            Matrix predict_label = predict(input.getRowAndTranspose(i));
            int max_label = 0;
            double max_predict = 0;
            for(int j = 0; j < predict_label.getHeight(); j++){
                if(predict_label.get(j,0) > max_predict) {
                    max_predict = predict_label.get(j,0);
                    max_label = j;
                }
            }
            if(max_label == (int)lables.get(i,0)) correctCount++;
        }
        System.out.println("accuracy: " + (double)correctCount / input.getHeight());
    }


    public static void main(String[] args) {
        List<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(784);
        layerSizes.add(100);
        layerSizes.add(10);

        layers = new ArrayList<>();
        for(int i = 0; i < layerSizes.size() - 1; i++){
            layers.add(new FullyConnectLayer(layerSizes.get(i), layerSizes.get(i+1), Sigmod::forward, Sigmod::backward));
        }

        String train_image_file = "Data\\MINST\\train-images.idx3-ubyte";
        String train_label_file = "Data\\MINST\\train-labels.idx1-ubyte";
        String test_image_file = "Data\\MINST\\t10k-images.idx3-ubyte";
        String test_label_file = "Data\\MINST\\t10k-labels.idx1-ubyte";

        Data.MINST.MNISTDataset train_dataset = new Data.MINST.MNISTDataset(train_image_file, train_label_file);
        Data.MINST.MNISTDataset test_dataset = new Data.MINST.MNISTDataset(test_image_file, test_label_file);

        int train_count = 1000;
        BatchData train_batch = train_dataset.getBatch(0, train_count);
        Matrix train_input = new Matrix(train_batch.images);
        Matrix train_lables = new Matrix(train_count, 1);
        for(int i = 0; i < train_count; i++){
            train_lables.set(i, 0, train_batch.labels[i]);
        }

        int test_count = 1000;
        BatchData test_batch = test_dataset.getBatch(0, test_count);
        Matrix test_input = new Matrix(test_batch.images);
        Matrix test_lables = new Matrix(test_count, 1);
        for(int i = 0; i < test_count; i++){
            test_lables.set(i, 0, test_batch.labels[i]);
        }

        train(train_input, train_lables, 0.001, 100);

        test(test_input, test_lables);

    }
}
