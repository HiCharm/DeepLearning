package source.model.FullyConnect;

import java.util.List;
import java.util.ArrayList;
import source.unit.Matrix;
import source.util.MatrixMath;
import source.util.VectorMath;

public class NetWork {
    public List<FullConnectedLayer> layers;
    List<Integer> layerSizes;

    public NetWork(List<Integer> layerSizes){
        layers = new ArrayList<>();
        for(int i = 0; i < layerSizes.size() - 1; i++){
            layers.add(new FullConnectedLayer(layerSizes.get(i), layerSizes.get(i+1), source.util.Sigmod::forward, source.util.Sigmod::backward));
        }
    }

    public Matrix predict(Matrix input) {
        Matrix output = input;
        for (FullConnectedLayer layer : layers) {
            layer.forward(output);
            output = layer.output;
        }
        return output;
    }

    public void update_weights(double learningRate) {
        for (FullConnectedLayer layer : layers) {
            layer.update(learningRate);
        }
    }

    public void calc_gradient(Matrix labels){
        FullConnectedLayer outputLayer = layers.get(layers.size() - 1);
        // 计算输出层的delta
        Matrix delta = new Matrix(1, outputLayer.outputSize);

        for(int i = 0; i < delta.getHeight(); i++){
            delta.set(i,0, (labels.get(i,0) - outputLayer.output.get(i,0)) * outputLayer.activationBackward.apply(outputLayer.output.get(i,0)));        
        }
        for(int i = layers.size() - 1; i >= 0; i--){
            layers.get(i).backward(delta);
            delta = layers.get(i).delta;
        }
    }

    public boolean train_one_sample(Matrix input, Matrix lables, double learningRate) {
        // 前向传播
        double predict_label = MatrixMath.max(predict(input), 0, true, true,true);
        // 计算梯度
        calc_gradient(lables);
        // 更新权重
        update_weights(learningRate);
        return predict_label == MatrixMath.max(lables, 0, true, true,true);
    }

    public void train(Matrix input, Matrix lables, double learningRate, int epoch) {
        for (int i = 0; i < epoch; i++) {
            System.out.println("epoch: " + i);
            int correctCount = 0;
            for(int j = 0; j < input.getHeight(); j++){
                Matrix sample_input = VectorMath.vector2Matrix(input.getRow(j), true);
                Matrix sample_lables = VectorMath.vector2Matrix(lables.getRow(j), true);
                if(train_one_sample(sample_input, sample_lables, learningRate)){
                    correctCount++;
                }
                if(j % 100 == 0){
                    System.out.println("\ntrained sample count: " + j);
                    System.out.println("current accuracy: " + (double)correctCount / (j + 1) );
                }
            }
        }
    }
}
