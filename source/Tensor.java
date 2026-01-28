package source;

import java.util.ArrayList;
import java.util.List;

public class Tensor {
    public List<Matrix> data;

    public Tensor(int depth, int height, int width) {
        data = new ArrayList<>();
        for (int i = 0; i < depth; i++) {
            data.add(new Matrix(height, width));
        }
    }

    public Tensor(List<Matrix> data) {
        this.data = data;
    }

    public int getDepth() {
        return data.size();
    }

    public int getHeight(int depth) {
        return data.get(depth).getHeight();
    }

    public int getWidth(int depth) {
        return data.get(depth).getWidth();
    }

    public Matrix get(int depth) {
        return data.get(depth);
    }

    public void set(int depth, int row, int col, double value) {
        data.get(depth).set(row, col, value);
    }

    public double get(int depth, int row, int col) {
        return data.get(depth).get(row, col);
    }

    public void set(int depth, Matrix value) {
        data.set(depth, value);
    }

    public void add(Tensor other) {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).add(other.get(i));
        }
    }

    public void add(double scalar) {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).add(scalar);
        }
    }

    public void subtract(Tensor other) {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).subtract(other.get(i));
        }
    }

    public void subtract(double scalar) {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).subtract(scalar);
        }
    }

    public void multiply(double scalar) {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).multiply(scalar);
        }
    }

    public void randomize() {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).randomize();
        }
    }

    public void fill(double value) {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).fill(value);
        }
    }

}
