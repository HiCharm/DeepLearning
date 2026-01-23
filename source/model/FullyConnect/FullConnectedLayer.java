package source.model.FullyConnect;

import java.util.function.Function;

import source.unit.Matrix;
import source.util.AdvanceMatrixMath;
import source.util.MatrixMath;


public class FullConnectedLayer {
    int inputSize;
    int outputSize;
    
    Function<Double, Double> activationFroward;
    Function<Double, Double> activationBackward;

    public Matrix input;
    public Matrix output;
    public Matrix weights;
    public Matrix bias;
    public Matrix delta;

    Matrix weights_grad;
    Matrix bias_grad;

    FullConnectedLayer(int inputSize, int outputSize, Function<Double, Double> activation_forward,Function<Double, Double> activation_backward) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationFroward = activation_forward;
        this.activationBackward = activation_backward;
        this.input = null;
        // 列向量 大小是 o * 1
        this.output = new Matrix(1, outputSize);
        MatrixMath.fillMatrix(0, output);
        // 权重矩阵 大小是 o * i
        this.weights = new Matrix(inputSize, outputSize); 
        MatrixMath.randomMatrix(-1*0.1, 0.1, weights);
        // 权重梯度矩阵 大小是 o * i
        this.weights_grad = new Matrix(inputSize, outputSize);
        MatrixMath.fillMatrix(0, weights_grad);
        // 偏置矩阵 大小是 o * 1    
        this.bias_grad = new Matrix(1, outputSize);
        MatrixMath.fillMatrix(0, bias_grad);
        // 列向量 大小是 o * 1
        this.bias = new Matrix(1, outputSize);
        MatrixMath.fillMatrix(0, bias);
        // 列向量 大小是 i * 1
        this.delta = new Matrix(1, inputSize);
    }

    public void forward(Matrix input) {
        // 输入大小是 i * 1 列向量
        this.input = input;

        /**
         * 矩阵乘法
         * input 是 i * 1 列向量
         * weights 是 o * i 矩阵
         */
        AdvanceMatrixMath.multiply(weights, input, output);
        MatrixMath.add(output, bias, true);
        // 应用激活函数
        for (int i = 0; i < output.getHeight(); i++) {
            for (int j = 0; j < output.getWidth(); j++) {
                output.set(i,j, activationFroward.apply(output.get(i,j)));
            }
        }
    }

    public void backward(Matrix delta_array){
        /**
         * 矩阵乘法
         * delta_array 是 o * 1 列向量
         * input 是 i * 1 列向量
         * input_T 是 1 * i 矩阵
         * weights_grad 是 o * i 矩阵
         */

        AdvanceMatrixMath.multiply(delta_array, MatrixMath.transpose(input), weights_grad);
        bias_grad = delta_array;
        // 计算 delta
        /**
         * 矩阵乘法
         * delta_array 是 o * 1 列向量
         * weights_T 是 i * o 矩阵
         * delta 是 i * 1 列向量
         * input 是 i * 1 列向量
         */
        AdvanceMatrixMath.multiply(MatrixMath.transpose(weights), delta_array, delta);
        for(int i = 0; i < delta.getHeight(); i++){
            for(int j = 0; j < delta.getWidth(); j++){
                delta.set(i, j, delta.get(i, j) * activationBackward.apply(input.get(i, j)));
            }
        }

    }

    public void update(double learningRate) {
        // 更新权重
        MatrixMath.multiply(learningRate, weights_grad);
        MatrixMath.add(weights, weights_grad, true);
        // 更新偏置
        MatrixMath.multiply(learningRate, bias_grad);
        MatrixMath.add(bias, bias_grad, true);
    }
}
