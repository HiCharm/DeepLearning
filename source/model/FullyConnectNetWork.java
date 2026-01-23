package source.model;

import java.util.ArrayList;
import java.util.List;

import Data.MINST.MNISTDataset.BatchData;
import source.unit.Matrix;
import source.util.MatrixMath;
import source.util.VectorMath;
import source.model.FullyConnect.NetWork;


public class FullyConnectNetWork extends NetWork {

    FullyConnectNetWork(List<Integer> layerSizes) {
        super(layerSizes);
    }

    public static void main(String[] args) {
        String train_image_file = "Data\\MINST\\train-images.idx3-ubyte";
        String train_label_file = "Data\\MINST\\train-labels.idx1-ubyte";
        String test_image_file = "Data\\MINST\\t10k-images.idx3-ubyte";
        String test_label_file = "Data\\MINST\\t10k-labels.idx1-ubyte";

        Data.MINST.MNISTDataset train_dataset = new Data.MINST.MNISTDataset(train_image_file, train_label_file);
        Data.MINST.MNISTDataset test_dataset = new Data.MINST.MNISTDataset(test_image_file, test_label_file);

        // 构建矩阵
        // int train_image_count = train_dataset.getImageCount();
        int train_image_count = 1000;

        BatchData batch = train_dataset.getBatch(0, train_image_count);
        Matrix train_images = new Matrix(batch.images);
        Matrix train_labels = new Matrix(10, train_image_count);
        for(int i = 0; i < train_image_count; i++){
            train_labels.set(i, batch.labels[i], 1.0);
        }

        //int test_image_count = test_dataset.getImageCount();
        int test_image_count = 1000;
        BatchData test_batch = test_dataset.getBatch(0, test_image_count);
        Matrix test_images = new Matrix(test_batch.images);
        Matrix test_labels = new Matrix(10, test_image_count);
        for(int i = 0; i < test_image_count; i++){
            test_labels.set(i, test_batch.labels[i], 1.0);
        }

        // 构建网络
        List<Integer> layerSizes = new ArrayList<>();
        layerSizes.add(train_images.getWidth());
        layerSizes.add(500);
        layerSizes.add(200);
        layerSizes.add(100);
        layerSizes.add(10);
        FullyConnectNetWork netWork = new FullyConnectNetWork(layerSizes);

        System.out.println("layersSize:" + netWork.layers.size());

        // 训练网络
        netWork.train(train_images, train_labels, 0.0001, 1);

        // 测试网络
        int correctCount = 0;
        for(int i = 0; i < test_image_count; i++){
            if(i % 100 == 0){
                System.out.println("\n测试第 " + i + " 个样本");
            }
            Matrix output = netWork.predict(VectorMath.vector2Matrix(test_images.getRow(i), true));
            int predictLabel = (int)MatrixMath.max(output, 0, true, true,true);
            int trueLabel = (int)MatrixMath.max(test_labels, i, false, true,true);
            if(predictLabel == trueLabel){
                correctCount++;
            }
        }
        System.out.println("正确率: " + (double)correctCount/test_image_count);

    }
    
}
