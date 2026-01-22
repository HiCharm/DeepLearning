package source.unit;

import java.util.ArrayList;
import java.util.List;

/**
 *  @HiCharm
 *  @Date: 2026-01-21 21:04
 *  @Description: 该矩阵类，仅作为数据存储单元
 */
public class Matrix {
    public List<Vector> rows = new ArrayList<>();

    /** 
     * 默认构造函数
     * 行数为0，列数为0
     */
    public Matrix() {   
    }

    /** 
     * 指定行数和列数的构造函数
     */
    public Matrix(int width, int height) {
        for (int i = 0; i < height; i++) {
            Vector row = new Vector(width);
            rows.add(row);
        }
    }

    /**
     * 使用数组初始化的矩阵
     */
    public Matrix(double[][] array) {
        for (double[] rowArray : array) {
            Vector row = new Vector(rowArray);
            rows.add(row);
        }
    }

    /**
     * 拷贝构造函数
     */
    public Matrix(Matrix matrix) {
        for (Vector row : matrix.rows) {
            Vector newRow = new Vector(row);
            rows.add(newRow);
        }
    }

    /**
     * 复制传入函数
     */
    public void copy(Matrix matrix) {
        this.rows.clear();
        for (Vector row : matrix.rows) {
            Vector newRow = new Vector(row);
            this.rows.add(newRow);
        }
    }
    /**
     * 复制传出函数
     */
    public Matrix copy() {
        Matrix newMatrix = new Matrix();
        for (Vector row : this.rows) {
            Vector newRow = new Vector(row);
            newMatrix.rows.add(newRow);
        }
        return newMatrix;
    }

    /**
     * 获取行数
     */
    public int getHeight() {
        return rows.size();
    }

    /**
     * 获取列数
     */
    public int getWidth() {
        if (rows.size() == 0) {
            return 0;
        }
        return rows.get(0).size();
    }

    /**
     * 获取指定行
     */
    public Vector getRow(int index) {
        return rows.get(index);
    }

    /**
     * 获取指定列
     */
    public Vector getColumn(int index) {
        Vector column = new Vector(getHeight());
        for (int i = 0; i < getHeight(); i++) {
            column.set(i, getRow(i).get(index));
        }
        return column;
    }

    /**
     * 设置指定行
     */
    public void setRow(int index, Vector row) {
        rows.set(index, row);
    }

    /**
     * 设置指定列
     */
    public void setColumn(int index, Vector column) {
        for (int i = 0; i < getHeight(); i++) {
            getRow(i).set(index, column.get(i));
        }
    }

    /**
     * 设置指定坐标的值
     */
    public void set(int rowIndex, int colIndex, double value) {
        rows.get(rowIndex).set(colIndex, value);
    }

    /**
     * 获取指定坐标的值
     */
    public double get(int rowIndex, int colIndex) {
        return rows.get(rowIndex).get(colIndex);
    }

    /**
     * 将某一列tostring
     */
    public String columnToString(int colIndex) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < getHeight(); i++) {
            sb.append(getRow(i).get(colIndex));
            if (i < getHeight() - 1) {
                sb.append(", ");
            }
        }
        return sb.toString();
    }

    /**
     * 将某一行tostring
     */
    public String rowToString(int rowIndex) {
        return rows.get(rowIndex).toString();
    }

    /**
     * 将矩阵tostring
     */
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < getHeight(); i++) {
            sb.append(rowToString(i));
            if (i < getHeight() - 1) {
                sb.append("\n");
            }
        }
        return sb.toString();
    }
}
