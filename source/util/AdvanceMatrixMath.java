package source.util;

import source.unit.Matrix;
import source.unit.Vector;

/**
 *  @HiCharm
 *  @Date: 2026-01-21 22:20
 *  @Description: 该高级矩阵数学计算类，实现更复杂的矩阵计算
 */
public class AdvanceMatrixMath {
    /**
     * 矩阵乘法，不覆盖
     */
    public static void multiply(Matrix m1, Matrix m2, Matrix result) {
        if (m1.getWidth() != m2.getHeight()) {
            System.out.println("m1 width: " + m1.getWidth() + ", m2 height: " + m2.getHeight());
            throw new IllegalArgumentException("Matrix multiplication not possible: m1 width must equal m2 height.");
        }
        if (result.getHeight() != m1.getHeight() || result.getWidth() != m2.getWidth()) {
            throw new IllegalArgumentException("Result matrix has incorrect dimensions for multiplication.");
        }
        for (int i = 0; i < m1.getHeight(); i++) {
            for (int j = 0; j < m2.getWidth(); j++) {
                double sum = 0.0;
                for (int k = 0; k < m1.getWidth(); k++) {
                    sum += m1.rows.get(i).get(k) * m2.rows.get(k).get(j);
                }
                result.rows.get(i).set(j, sum);
            }
        }
    }

    /**
     * 矩阵乘法，覆盖
     */
    public static void multiply(Matrix m1, Matrix m2, boolean left){
        if (left) {
            if (m1.getWidth() != m2.getHeight()) {
                throw new IllegalArgumentException("Matrix multiplication not possible: m1 width must equal m2 height.");
            }
            for (int i = 0; i < m1.getHeight(); i++) {
                for (int j = 0; j < m2.getWidth(); j++) {
                    double sum = 0.0;
                    for (int k = 0; k < m1.getWidth(); k++) {
                        sum += m1.rows.get(i).get(k) * m2.rows.get(k).get(j);
                    }
                    m1.rows.get(i).set(j, sum);
                }
            }
        } else {
            if (m1.getHeight() != m2.getWidth()) {
                throw new IllegalArgumentException("Matrix multiplication not possible: m1 height must equal m2 width.");
            }
            for (int i = 0; i < m1.getHeight(); i++) {
                for (int j = 0; j < m2.getWidth(); j++) {
                    double sum = 0.0;
                    for (int k = 0; k < m1.getWidth(); k++) {
                        sum += m1.rows.get(i).get(k) * m2.rows.get(k).get(j);
                    }
                    m2.rows.get(i).set(j, sum);
                }
            }
        }
    }
    
    /**
     * 使用高斯消元法计算行列式
     */
    public static double determinant(Matrix matrix) {
        if (matrix.getHeight() != matrix.getWidth()) {
            throw new IllegalArgumentException("Determinant can only be calculated for square matrices.");
        }
        int n = matrix.getHeight();
        Matrix tempMatrix = matrix.copy();
        double det = 1.0;

        for (int i = 0; i < n; i++) {
            // 寻找主元
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(tempMatrix.rows.get(k).get(i)) > Math.abs(tempMatrix.rows.get(maxRow).get(i))) {
                    maxRow = k;
                }
            }
            // 交换行
            if (i != maxRow) {
                Vector tempRow = tempMatrix.rows.get(i);
                tempMatrix.rows.set(i, tempMatrix.rows.get(maxRow));
                tempMatrix.rows.set(maxRow, tempRow);
                det = -det; // 行交换改变行列式符号
            }
            // 主元为0，行列式为0
            if (Math.abs(tempMatrix.rows.get(i).get(i)) < 1e-10) {
                return 0.0;
            }
            det *= tempMatrix.rows.get(i).get(i);
            // 消元
            for (int k = i + 1; k < n; k++) {
                double factor = tempMatrix.rows.get(k).get(i) / tempMatrix.rows.get(i).get(i);
                for (int j = i; j < n; j++) {
                    double value = tempMatrix.rows.get(k).get(j) - factor * tempMatrix.rows.get(i).get(j);
                    tempMatrix.rows.get(k).set(j, value);
                }
            }
        }
        return det;
    }

    /**
     * 计算矩阵的迹
     */
    public static double trace(Matrix matrix) {
        if (matrix.getHeight() != matrix.getWidth()) {
            throw new IllegalArgumentException("Trace can only be calculated for square matrices.");
        }
        double trace = 0.0;
        for (int i = 0; i < matrix.getHeight(); i++) {
            trace += matrix.rows.get(i).get(i);
        }
        return trace;
    }

    /**
     * 计算矩阵的秩
     */
    public static int rank(Matrix matrix) {
        int n = matrix.getHeight();
        int rank = 0;
        for (int i = 0; i < n; i++) {
            boolean zeroRow = true;
            for (int j = 0; j < n; j++) {
                if (Math.abs(matrix.rows.get(i).get(j)) > 1e-10) {
                    zeroRow = false;
                    break;
                }
            }
            if (!zeroRow) {
                rank++;
            }
        }
        return rank;
    }

    /**
     * 获取余子式矩阵
     */
    public static Matrix getCofactorMatrix(Matrix matrix, int row, int col) {
        int n = matrix.getHeight();
        Matrix cofactorMatrix = new Matrix(n - 1, n - 1);
        int r = 0;
        for (int i = 0; i < n; i++) {
            if (i == row) continue;
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (j == col) continue;
                cofactorMatrix.set(r, c, matrix.rows.get(i).get(j));
                c++;
            }
            r++;
        }
        return cofactorMatrix;
    }

    /**
     * 计算矩阵的逆矩阵
     */
    public static Matrix inverse(Matrix matrix) {
        if (matrix.getHeight() != matrix.getWidth()) {
            throw new IllegalArgumentException("Inverse can only be calculated for square matrices.");
        }
        int n = matrix.getHeight();
        Matrix tempMatrix = matrix.copy();
        Matrix inverseMatrix = new Matrix(n, n);
        double det = determinant(tempMatrix);
        if (Math.abs(det) < 1e-10) {
            throw new ArithmeticException("Matrix is not invertible.");
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double factor = (i + j) % 2 == 0 ? 1.0 : -1.0;
                double value = factor * determinant(getCofactorMatrix(tempMatrix, i, j));
                inverseMatrix.set(i, j, value / det);
            }
        }
        return inverseMatrix;
    }

    

}
