package source.Layer;

import source.Matrix;
import source.Tensor;

public class MaxPoolingLayer {
    public int inputHeight;
    public int inputWidth;
    public int channel;
    public int filterHeight;
    public int filterWidth;
    public int stride;
    public int outputHeight;
    public int outputWidth;
    
    public Tensor input;
    public Tensor output;
    public Tensor deltaArray;
    
    public Tensor maxIndices_x;
    public Tensor maxIndices_y;
    
    public MaxPoolingLayer(int inputHeight, int inputWidth, int channel, 
                          int filterHeight, int filterWidth, int stride) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.channel = channel;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        this.stride = stride;
        
        this.outputHeight = (inputHeight - filterHeight) / stride + 1;
        this.outputWidth = (inputWidth - filterWidth) / stride + 1;
        this.output = new Tensor(channel, outputHeight, outputWidth);
        this.maxIndices_x = new Tensor(channel, outputHeight, outputWidth);
        this.maxIndices_y = new Tensor(channel, outputHeight, outputWidth);
    }
    
    public Matrix getPatch(Tensor input, int channel, int i, int j, 
                          int filterWidth, int filterHeight, int stride) {
        int startRow = i * stride;
        int startCol = j * stride;
        int endRow = Math.min(startRow + filterHeight, inputHeight);
        int endCol = Math.min(startCol + filterWidth, inputWidth);
        
        Matrix patch = new Matrix(endRow - startRow, endCol - startCol);
        for (int r = startRow; r < endRow; r++) {
            for (int c = startCol; c < endCol; c++) {
                patch.set(r - startRow, c - startCol, input.get(channel, r, c));
            }
        }
        return patch;
    }
    
    public int[] getMaxIndex(Matrix patch) {
        double maxVal = -Double.MAX_VALUE;
        int maxRow = 0;
        int maxCol = 0;
        
        for (int i = 0; i < patch.getHeight(); i++) {
            for (int j = 0; j < patch.getWidth(); j++) {
                if (patch.get(i, j) > maxVal) {
                    maxVal = patch.get(i, j);
                    maxRow = i;
                    maxCol = j;
                }
            }
        }
        return new int[]{maxRow, maxCol};
    }
    
    public void forward(Tensor input) {
        this.input = input;
        
        for (int d = 0; d < channel; d++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    Matrix patch = getPatch(input, d, i, j, filterWidth, filterHeight, stride);
                    int[] maxIndex = getMaxIndex(patch);
                    
                    int absoluteRow = i * stride + maxIndex[0];
                    int absoluteCol = j * stride + maxIndex[1];
                    maxIndices_x.set(d, i, j, absoluteRow);
                    maxIndices_y.set(d, i, j, absoluteCol);
                    
                    output.set(d, i, j, input.get(d, absoluteRow, absoluteCol));
                }
            }
        }
    }
    
    public void backward(Tensor sensitivityArray) {
        this.deltaArray = new Tensor(channel, inputHeight, inputWidth);
        
        for (int d = 0; d < channel; d++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    int maxRow = (int) maxIndices_x.get(d, i, j);
                    int maxCol = (int) maxIndices_y.get(d, i, j);
                    
                    deltaArray.set(d, maxRow, maxCol, sensitivityArray.get(d, i, j));
                }
            }
        }
    }
}