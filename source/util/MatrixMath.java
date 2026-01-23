package source.util;

import source.unit.Matrix;
import source.unit.Vector;

/**
 *  @HiCharm
 *  @Date: 2026-01-21 21:38
 *  @Description: 该矩阵数学计算类，用于构建矩阵，以及矩阵计算
 */
public class MatrixMath {
    /**
     * 常数矩阵填充
     */
    public static void fillMatrix(double value, Matrix matrix){
        for (int i = 0; i < matrix.getHeight(); i++) {
            for (int j = 0; j < matrix.getWidth(); j++) {
                matrix.rows.get(i).set(j, value);
            }
        }
    }

    /**
     * 矩阵按行序列化填充
     */
    public static void rowMatrix(int start, int end, int step, Matrix matrix) {
        if(matrix.getWidth() != ((end - start) / step + 1)) {
            throw new IllegalArgumentException("Matrix width does not match the specified range and step.");
        }
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.rangeVector(start, end, step, matrix.rows.get(i));
        }
    }

    /**
     * 矩阵按列序列化填充
     */
    public static void columnMatrix(int start, int end, int step, Matrix matrix) {
        if(matrix.getHeight() != ((end - start) / step + 1)) {
            throw new IllegalArgumentException("Matrix height does not match the specified range and step.");
        }
        for (int i = 0; i < matrix.getWidth(); i++) {
            Vector column = new Vector(matrix.getHeight());
            for (int j = 0; j < matrix.getHeight(); j++) {
                column.set(j, matrix.rows.get(j).get(i));
            }
            matrix.setColumn(i, column);
        }
    }

    /**
     * 矩阵的随机数填充
     */
    public static void randomMatrix(double min, double max, Matrix matrix){
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.randomVector(min, max, matrix.rows.get(i));
        }
    }

    /**
     * 矩阵加法，无返回值，不覆盖
     */
    public static void add(Matrix m1, Matrix m2, Matrix result) {
        if (m1.getHeight() != m2.getHeight() || m1.getWidth() != m2.getWidth()) {
            throw new IllegalArgumentException("Matrices must be of the same dimensions for addition.");
        }
        if (result.getHeight() != m1.getHeight() || result.getWidth() != m1.getWidth()) {
            throw new IllegalArgumentException("Result matrix must be of the same dimensions as the input matrices for addition.");
        }
        for (int i = 0; i < m1.getHeight(); i++) {
            VectorMath.add(m1.rows.get(i), m2.rows.get(i), result.rows.get(i));
        }
    }

    /**
     * 矩阵加法，无返回值，覆盖
     */
    public static void add(Matrix m1, Matrix m2, boolean left){
        if (left) {
            for (int i = 0; i < m1.getHeight(); i++) {
                VectorMath.add(m1.rows.get(i), m2.rows.get(i), true);
            }
        } else {
            for (int i = 0; i < m1.getHeight(); i++) {
                VectorMath.add(m1.rows.get(i), m2.rows.get(i), false);
            }
        }
    }

    /**
     * 矩阵减法，无返回值，不覆盖
     */
    public static void subtract(Matrix m1, Matrix m2, Matrix result) {
        if (m1.getHeight() != m2.getHeight() || m1.getWidth() != m2.getWidth()) {
            throw new IllegalArgumentException("Matrices must be of the same dimensions for subtraction.");
        }
        if (result.getHeight() != m1.getHeight() || result.getWidth() != m1.getWidth()) {
            throw new IllegalArgumentException("Result matrix must be of the same dimensions as the input matrices for subtraction.");
        }
        for (int i = 0; i < m1.getHeight(); i++) {
            VectorMath.subtract(m1.rows.get(i), m2.rows.get(i), result.rows.get(i));
        }
    }

    /**
     * 矩阵减法，无返回值，覆盖
     */
    public static void subtract(Matrix m1, Matrix m2, boolean left){
        if (left) {
            for (int i = 0; i < m1.getHeight(); i++) {
                VectorMath.subtract(m1.rows.get(i), m2.rows.get(i), true);
            }
        } else {
            for (int i = 0; i < m1.getHeight(); i++) {
                VectorMath.subtract(m1.rows.get(i), m2.rows.get(i), false);
            }
        }
    }

    /**
     * 矩阵数乘，无返回值，不覆盖
     */
    public static void multiply(double scalar, Matrix matrix, Matrix result) {
        if (result.getHeight() != matrix.getHeight() || result.getWidth() != matrix.getWidth()) {
            throw new IllegalArgumentException("Result matrix must be of the same dimensions as the input matrix for multiplication.");
        }
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.multiply(scalar, matrix.rows.get(i), result.rows.get(i));
        }
    }

    /**
     * 矩阵数乘，无返回值，覆盖
     */
    public static void multiply(double scalar, Matrix matrix){
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.multiply(scalar, matrix.rows.get(i));
        }
    }

    /**
     * 矩阵标量加，不覆盖
     */
    public static void scalarAdd(double scalar, Matrix matrix, Matrix result) {
        if (result.getHeight() != matrix.getHeight() || result.getWidth() != matrix.getWidth()) {
            throw new IllegalArgumentException("Result matrix must be of the same dimensions as the input matrix for scalar addition.");
        }
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.add(scalar, matrix.rows.get(i), result.rows.get(i));
        }
    }
    /**
     * 矩阵标量加，覆盖
     */
    public static void scalarAdd(double scalar, Matrix matrix){
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.add(scalar, matrix.rows.get(i));
        }
    }

    /**
     * 矩阵标量减，不覆盖
     */
    public static void scalarSubtract(double scalar, Matrix matrix, Matrix result) {
        if (result.getHeight() != matrix.getHeight() || result.getWidth() != matrix.getWidth()) {
            throw new IllegalArgumentException("Result matrix must be of the same dimensions as the input matrix for scalar subtraction.");
        }
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.subtract(scalar, matrix.rows.get(i), result.rows.get(i));
        }
    }

    /**
     * 矩阵标量减，覆盖
     */
    public static void scalarSubtract(double scalar, Matrix matrix){
        for (int i = 0; i < matrix.getHeight(); i++) {
            VectorMath.subtract(scalar, matrix.rows.get(i));
        }
    }

    /**
     * 矩阵转置，不覆盖
     */
    public static void transpose(Matrix matrix, Matrix result) {
        if (result.getHeight() != matrix.getWidth() || result.getWidth() != matrix.getHeight()) {
            throw new IllegalArgumentException("Result matrix must be of the dimensions of the transposed matrix.");
        }
        for (int i = 0; i < matrix.getHeight(); i++) {
            for (int j = 0; j < matrix.getWidth(); j++) {
                result.set(j, i, matrix.get(i, j));
            }
        }
    }

    /**
     * 矩阵转置,有返回值
     */
    public static Matrix transpose(Matrix matrix){
        Matrix result = new Matrix(matrix.getHeight(), matrix.getWidth());
        for (int i = 0; i < matrix.getHeight(); i++) {
            for (int j = 0; j < matrix.getWidth(); j++) {
                result.set(j, i, matrix.get(i, j));
            }
        }
        return result;
    }


    /**
     * 求矩阵某一列的和
     */
    public static double sum(Matrix matrix, int colIndex){
        double sum = 0.0;
        for (int i = 0; i < matrix.getHeight(); i++) {
            sum += matrix.get(i, colIndex);
        }
        return sum;
    }

    /**
     * 求矩阵某一列或某一行最大或最小的数
     */
    public static double max(Matrix matrix, int index, boolean isColumn, boolean isMax,boolean returnIndex){
        double extremeValue;
        if(isMax){
            extremeValue = Double.NEGATIVE_INFINITY;
        } else {
            extremeValue = Double.POSITIVE_INFINITY;
        }
        int extremeIndex = -1;
        if(isColumn){
            for (int i = 0; i < matrix.getHeight(); i++) {
                double value = matrix.get(i, index);
                if((isMax && value > extremeValue) || (!isMax && value < extremeValue)){
                    extremeValue = value;
                    extremeIndex = i;
                }
            }
        } else {
            for (int i = 0; i < matrix.getWidth(); i++) {
                double value = matrix.get(index, i);
                if((isMax && value > extremeValue) || (!isMax && value < extremeValue)){
                    extremeValue = value;
                    extremeIndex = i;
                }
            }
        }
        if(returnIndex){
            return extremeIndex;
        } else {
            return extremeValue;
        }
    }
}