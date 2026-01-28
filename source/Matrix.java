package source;

import java.util.List;
import java.util.function.Function;

public class Matrix {
    double[][] data;

    public Matrix(int height, int width) {
        data = new double[height][width];
        fill(0);
    }

    public Matrix(double[][] data) {
        this.data = data;
    }

    public Matrix(List<List<Double>> data) {
        this.data = new double[data.size()][data.get(0).size()];
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.get(0).size(); j++) {
                this.data[i][j] = data.get(i).get(j);
            }
        }
    }

    public int getHeight() {
        return data.length;
    }

    public int getWidth() {
        return data[0].length;
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    public void set(int row, int col, double value) {
        data[row][col] = value;
    }

    public void add(Matrix other) {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] += other.get(i, j);
            }
        }
    }

    public void add(double scalar) {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] += scalar;
            }
        }
    }

    public void subtract(double scalar) {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] -= scalar;
            }
        }
    }

    public void subtract(Matrix other) {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] -= other.get(i, j);
            }
        }
    }

    public void multiply(double scalar) {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] *= scalar;
            }
        }
    }

    public Matrix transpose() {
        Matrix transposed = new Matrix(getWidth(), getHeight());
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                transposed.set(j, i, data[i][j]);
            }
        }
        return transposed;
    }

    public Matrix copy() {
        Matrix copy = new Matrix(getHeight(), getWidth());
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                copy.set(i, j, data[i][j]);
            }
        }
        return copy;
    }

    public Matrix multiply(Matrix other) {
        if (this.getWidth() != other.getHeight()) {
            throw new IllegalArgumentException("Incompatible matrix sizes for multiplication.");
        }
        Matrix result = new Matrix(this.getHeight(), other.getWidth());
        for (int i = 0; i < this.getHeight(); i++) {
            for (int j = 0; j < other.getWidth(); j++) {
                double sum = 0;
                for (int k = 0; k < this.getWidth(); k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    public void randomize() {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] = Math.random();
            }
        }
    }

    public void fill(double value) {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] = value;
            }
        }
    }

    public Matrix getRowAndTranspose(int row) {
        Matrix rowVector = new Matrix(getWidth(), 1);
        for(int j = 0; j < getWidth(); j++)
            rowVector.set(j, 0, data[row][j]);
        return rowVector;
    }

    public void apply(Function<Double, Double> function) {
        for (int i = 0; i < getHeight(); i++) {
            for (int j = 0; j < getWidth(); j++) {
                data[i][j] = function.apply(data[i][j]);
            }
        }
    }

}
