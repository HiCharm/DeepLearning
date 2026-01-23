package Data.MINST;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * MNIST数据集读取器
 * 功能：读取MNIST图像和标签文件，提供按索引访问图像和标签的方法
 */
public class MNISTDataset {
    private double[][] images; // 存储所有图像的一维数组表示
    private int[] labels;     // 存储所有标签
    private int imageCount;   // 图像数量
    private int imageSize;    // 每张图像的像素数量(28×28=784)
    
    /**
     * 构造函数：读取MNIST数据集
     * @param imageFilePath 图像文件路径(如: train-images-idx3-ubyte)
     * @param labelFilePath 标签文件路径(如: train-labels-idx1-ubyte)
     */
    public MNISTDataset(String imageFilePath, String labelFilePath) {
        try {
            // 读取图像文件
            this.images = readImageFile(imageFilePath);
            // 读取标签文件
            this.labels = readLabelFile(labelFilePath);
            
            // 验证图像和标签数量是否匹配
            if (images.length != labels.length) {
                throw new IllegalArgumentException("图像数量(" + images.length + 
                                                 ")与标签数量(" + labels.length + ")不匹配");
            }
            
            this.imageCount = images.length;
            this.imageSize = (images.length > 0) ? images[0].length : 0;
            
            System.out.println("成功加载MNIST数据集:");
            System.out.println("图像数量: " + imageCount);
            System.out.println("每张图像尺寸: " + imageSize + "像素(28×28)");
            
        } catch (IOException e) {
            throw new RuntimeException("读取MNIST数据集失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 读取MNIST图像文件
     */
    private double[][] readImageFile(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            
            // 读取文件头信息[1,4](@ref)
            int magicNumber = dis.readInt();
            if (magicNumber != 2051) {
                throw new IOException("无效的MNIST图像文件，魔数应为2051，实际为: " + magicNumber);
            }
            
            int numberOfImages = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();
            int pixelCount = rows * cols;
            
            System.out.println("图像文件头信息:");
            System.out.println("图像数量: " + numberOfImages);
            System.out.println("每张图像像素数: " + pixelCount);
            
            // 读取所有图像数据[4](@ref)
            double[][] imageData = new double[numberOfImages][pixelCount];
            for (int i = 0; i < numberOfImages; i++) {
                for (int j = 0; j < pixelCount; j++) {
                    // 读取像素值并归一化到0-1范围[2,7](@ref)
                    int pixel = dis.readUnsignedByte(); // 使用无符号字节读取
                    imageData[i][j] = pixel / 255.0;    // 归一化处理
                }
                // 进度显示
                if ((i + 1) % 10000 == 0) {
                    System.out.println("已读取 " + (i + 1) + " 张图像");
                }
            }
            
            return imageData;
        }
    }
    
    /**
     * 读取MNIST标签文件
     */
    private int[] readLabelFile(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            
            // 读取文件头信息[1](@ref)
            int magicNumber = dis.readInt();
            if (magicNumber != 2049) {
                throw new IOException("无效的MNIST标签文件，魔数应为2049，实际为: " + magicNumber);
            }
            
            int numberOfLabels = dis.readInt();
            
            // 读取所有标签[4](@ref)
            int[] labelData = new int[numberOfLabels];
            for (int i = 0; i < numberOfLabels; i++) {
                labelData[i] = dis.readUnsignedByte(); // 标签值0-9
            }
            
            return labelData;
        }
    }
    
    /**
     * 根据索引获取单张图像的一维数组表示
     * @param index 图像索引(0到imageCount-1)
     * @return 归一化后的图像像素数组(0-1范围)
     */
    public double[] getImage(int index) {
        checkIndex(index);
        return images[index];
    }
    
    /**
     * 根据索引获取对应的标签
     * @param index 图像索引
     * @return 标签值(0-9)
     */
    public int getLabel(int index) {
        checkIndex(index);
        return labels[index];
    }
    
    /**
     * 批量获取图像和标签
     * @param startIndex 起始索引
     * @param count 要获取的数量
     * @return 包含图像和标签的批次数据
     */
    public BatchData getBatch(int startIndex, int count) {
        checkIndex(startIndex);
        if (startIndex + count > imageCount) {
            throw new IllegalArgumentException("请求的批次超出数据集范围");
        }
        
        double[][] batchImages = new double[count][imageSize];
        int[] batchLabels = new int[count];
        
        for (int i = 0; i < count; i++) {
            batchImages[i] = images[startIndex + i];
            batchLabels[i] = labels[startIndex + i];
        }
        
        return new BatchData(batchImages, batchLabels);
    }
    
    /**
     * 获取所有图像的索引列表
     * @return 0到imageCount-1的索引列表
     */
    public List<Integer> getAllIndices() {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < imageCount; i++) {
            indices.add(i);
        }
        return indices;
    }
    
    /**
     * 根据标签值筛选图像索引
     * @param targetLabel 目标标签(0-9)
     * @return 符合标签的所有图像索引
     */
    public List<Integer> getIndicesByLabel(int targetLabel) {
        if (targetLabel < 0 || targetLabel > 9) {
            throw new IllegalArgumentException("标签值应在0-9范围内");
        }
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < imageCount; i++) {
            if (labels[i] == targetLabel) {
                indices.add(i);
            }
        }
        return indices;
    }
    
    /**
     * 获取数据集统计信息
     */
    public void printStatistics() {
        System.out.println("\n数据集统计信息:");
        System.out.println("总图像数量: " + imageCount);
        System.out.println("图像尺寸: " + imageSize + "像素");
        
        // 统计每个标签的数量[7](@ref)
        int[] labelCounts = new int[10];
        for (int label : labels) {
            labelCounts[label]++;
        }
        
        System.out.println("标签分布:");
        for (int i = 0; i < 10; i++) {
            System.out.println("数字 " + i + ": " + labelCounts[i] + "张图像 (" + 
                             String.format("%.1f", labelCounts[i] * 100.0 / imageCount) + "%)");
        }
    }
    
    /**
     * 验证索引有效性
     */
    private void checkIndex(int index) {
        if (index < 0 || index >= imageCount) {
            throw new IllegalArgumentException("索引超出范围: " + index + 
                                             ", 有效范围: 0-" + (imageCount - 1));
        }
    }
    
    // Getter方法
    public int getImageCount() { return imageCount; }
    public int getImageSize() { return imageSize; }
    public double[][] getAllImages() { return images; }
    public int[] getAllLabels() { return labels; }
    
    /**
     * 批次数据容器类
     */
    public static class BatchData {
        public final double[][] images;
        public final int[] labels;
        
        public BatchData(double[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
        
        public int size() { return images.length; }
    }
    
    /**
     * 使用示例
     */
    public static void main(String[] args) {
        // 示例用法
        String imageFile = "Data\\MINST\\train-images.idx3-ubyte";
        String labelFile = "Data\\MINST\\train-labels.idx1-ubyte";
        
        try {
            MNISTDataset dataset = new MNISTDataset(imageFile, labelFile);
            
            // 打印统计信息
            dataset.printStatistics();
            
            // 示例1：获取单张图像和标签
            System.out.println("\n示例1：获取前5张图像和标签:");
            for (int i = 0; i < 5; i++) {
                double[] image = dataset.getImage(i);
                int label = dataset.getLabel(i);
                System.out.println("图像" + i + ": 标签=" + label + 
                                 ", 像素数量=" + image.length + 
                                 ", 第一个像素值=" + String.format("%.4f", image[0]));
            }
            
            // 示例2：批量获取图像
            System.out.println("\n示例2：批量获取图像(索引10-14):");
            BatchData batch = dataset.getBatch(10, 5);
            for (int i = 0; i < batch.size(); i++) {
                System.out.println("批次图像" + i + ": 标签=" + batch.labels[i]);
            }
            
            // 示例3：按标签筛选
            System.out.println("\n示例3：数字5的图像索引(前10个):");
            List<Integer> indicesOf5 = dataset.getIndicesByLabel(5);
            for (int i = 0; i < Math.min(10, indicesOf5.size()); i++) {
                int idx = indicesOf5.get(i);
                System.out.println("索引" + idx + ": 标签=" + dataset.getLabel(idx));
            }
            
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
