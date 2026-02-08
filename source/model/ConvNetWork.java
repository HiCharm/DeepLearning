package source.model;

import java.util.ArrayList;
import java.util.List;

import Data.MNIST.MNISTDataset;
import Data.MNIST.MNISTDataset.BatchData;
import source.Layer.ConvLayer;
import source.Layer.FullyConnectLayer;
import source.Layer.MaxPoolingLayer;
import source.Matrix;
import source.Tensor;
import util.Relu;

public class ConvNetWork {

	static ConvLayer conv;
	static MaxPoolingLayer pool;
	static List<FullyConnectLayer> fcLayers;

	public static Matrix predict(Tensor input) {
		// conv -> pool -> flatten -> fully connected
		conv.forward(input);
		// keep padded input for gradient calc
		conv.padded_input = conv.padding(input, conv.zp);

		pool.forward(conv.output);

		Matrix fcInput = tensorToColumnMatrix(pool.output);
		Matrix output = fcInput;
		for (int i = 0; i < fcLayers.size(); i++) {
			fcLayers.get(i).forward(output);
			output = fcLayers.get(i).output;
		}
		return output;
	}

	public static void update_weights(double learningRate) {
		for (int i = 0; i < fcLayers.size(); i++) {
			fcLayers.get(i).update(learningRate);
		}
		conv.update();
	}

	public static void calc_gradient(Matrix labels) {
		FullyConnectLayer outputLayer = fcLayers.get(fcLayers.size() - 1);
		// apply softmax on linear outputs and use cross-entropy gradient: delta = softmax - labels
		int n = outputLayer.output.getHeight();
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			if (outputLayer.output.get(i, 0) > max) max = outputLayer.output.get(i, 0);
		}
		double sumExp = 0.0;
		double[] exps = new double[n];
		for (int i = 0; i < n; i++) {
			exps[i] = Math.exp(outputLayer.output.get(i, 0) - max);
			sumExp += exps[i];
		}
		Matrix delta = new Matrix(n, 1);
		for (int i = 0; i < n; i++) {
			double soft = exps[i] / sumExp;
			delta.set(i, 0, soft - labels.get(i, 0));
		}

		// backprop through fully connected layers
		for (int i = fcLayers.size() - 1; i >= 0; i--) {
			fcLayers.get(i).backward(delta);
			delta = fcLayers.get(i).delta;
		}

		// delta now is gradient wrt flattened pooling output
		// reshape to tensor matching pool.output (depth, h, w)
		int depth = pool.output.getDepth();
		int h = pool.output.getHeight(0);
		int w = pool.output.getWidth(0);
		Tensor poolDelta = columnMatrixToTensor(delta, depth, h, w);

		// backprop through pooling
		pool.backward(poolDelta);

		// backprop through conv
		// initialize weight grads for filters
		for (int i = 0; i < conv.filterNumber; i++) {
			ConvLayer.Filter filter = conv.filters.get(i);
			filter.weights_grad = new Tensor(conv.channel, conv.filterHeight, conv.filterWidth);
		}
		// ensure padded_input is set (forward saved input but not padded_input)
		conv.padded_input = conv.padding(conv.input, conv.zp);

		// compute gradients for conv filters
		conv.bpGradient(pool.deltaArray);
		// compute delta wrt conv input if needed (not used further here)
		conv.bpDeltaMap(pool.deltaArray);
	}

	public static boolean train_one_sample(Matrix inputColumn, int label, double learningRate) {
		// convert column vector to tensor 1x28x28
		Tensor input = columnMatrixToImageTensor(inputColumn, 28, 28);
		Matrix predict_label = predict(input);

		Matrix labels = new Matrix(10, 1);
		labels.fill(0);
		labels.set(label, 0, 1.0);

		calc_gradient(labels);
		update_weights(learningRate);

		double max_predict = Double.NEGATIVE_INFINITY;
		int max_label = 0;
		for (int i = 0; i < predict_label.getHeight(); i++) {
			if (predict_label.get(i, 0) > max_predict) {
				max_predict = predict_label.get(i, 0);
				max_label = i;
			}
		}
        // System.out.println("predict: " + max_label + " label: " + label);
		return max_label == label;
	}

	public static void train(Matrix input, Matrix labels, double learningRate, int epoch) {
		for (int e = 0; e < epoch; e++) {
			System.out.println("epoch: " + e);
			int correctCount = 0;
			for (int i = 0; i < input.getHeight(); i++) {
				if (train_one_sample(input.getRowAndTranspose(i), (int) labels.get(i, 0), learningRate)) {
					correctCount++;
				}
				if (i % 100 == 0) {
					System.out.println("trained sample count: " + i + " current accuracy: " + (double) correctCount / (i + 1));
				}
			}
		}
	}

	public static void test(Matrix input, Matrix labels) {
		int correctCount = 0;
		for (int i = 0; i < input.getHeight(); i++) {
			Tensor in = columnMatrixToImageTensor(input.getRowAndTranspose(i), 28, 28);
			Matrix predict_label = predict(in);
			int max_label = 0;
			double max_predict = Double.NEGATIVE_INFINITY;
			for (int j = 0; j < predict_label.getHeight(); j++) {
				if (predict_label.get(j, 0) > max_predict) {
					max_predict = predict_label.get(j, 0);
					max_label = j;
				}
			}
			if (max_label == (int) labels.get(i, 0)) correctCount++;
		}
		System.out.println("accuracy: " + (double) correctCount / input.getHeight());
	}

	// helper: flatten tensor to column matrix
	public static Matrix tensorToColumnMatrix(Tensor t) {
		int d = t.getDepth();
		int h = t.getHeight(0);
		int w = t.getWidth(0);
		Matrix m = new Matrix(d * h * w, 1);
		int idx = 0;
		for (int depth = 0; depth < d; depth++) {
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					m.set(idx++, 0, t.get(depth, i, j));
				}
			}
		}
		return m;
	}

	public static Tensor columnMatrixToTensor(Matrix m, int depth, int h, int w) {
		Tensor t = new Tensor(depth, h, w);
		int idx = 0;
		for (int d = 0; d < depth; d++) {
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					t.set(d, i, j, m.get(idx++, 0));
				}
			}
		}
		return t;
	}

	public static Tensor columnMatrixToImageTensor(Matrix m, int height, int width) {
		Tensor t = new Tensor(1, height, width);
		int idx = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				t.set(0, i, j, m.get(idx++, 0));
			}
		}
		return t;
	}

	public static void main(String[] args) {
		conv = new ConvLayer(28, 28, 1, 3, 3, 8, 1, 1, 0.01, Relu::forward, Relu::backward);
		pool = new MaxPoolingLayer(conv.outputHeight, conv.outputWidth, conv.filterNumber, 2, 2, 2);

		int flattened = pool.output.getDepth() * pool.output.getHeight(0) * pool.output.getWidth(0);

        System.out.println("flattened: " + flattened);

		fcLayers = new ArrayList<>();
		fcLayers.add(new FullyConnectLayer(flattened, 100, Relu::forward, Relu::backward));
		// last layer linear (identity) so we can apply softmax + cross-entropy
		fcLayers.add(new FullyConnectLayer(100, 10, x -> x, x -> 1.0));

		String train_image_file = "Data\\MINST\\train-images.idx3-ubyte";
		String train_label_file = "Data\\MINST\\train-labels.idx1-ubyte";
		String test_image_file = "Data\\MINST\\t10k-images.idx3-ubyte";
		String test_label_file = "Data\\MINST\\t10k-labels.idx1-ubyte";

		MNISTDataset train_dataset = new MNISTDataset(train_image_file, train_label_file);
		MNISTDataset test_dataset = new MNISTDataset(test_image_file, test_label_file);

		int train_count = 1000;
		BatchData train_batch = train_dataset.getBatch(0, train_count);
		Matrix train_input = new Matrix(train_batch.images);
		Matrix train_labels = new Matrix(train_count, 1);
		for (int i = 0; i < train_count; i++) train_labels.set(i, 0, train_batch.labels[i]);

		int test_count = 1000;
		BatchData test_batch = test_dataset.getBatch(0, test_count);
		Matrix test_input = new Matrix(test_batch.images);
		Matrix test_labels = new Matrix(test_count, 1);
		for (int i = 0; i < test_count; i++) test_labels.set(i, 0, test_batch.labels[i]);

		train(train_input, train_labels, 0.01, 5);
		test(test_input, test_labels);
	}

}
