package source.util;

import source.unit.Vector;

/**
 *  @HiCharm
 *  @Date: 2026-01-21 21:23
 *  @Description: 该向量数学计算类，用于构建向量，以及向量计算
 */
public class VectorMath {
    /**
     * 向量的range序列化生成
     */
    public static void rangeVector(int start, int end, int step, Vector vector) {
        if(vector.size() != ((end - start) / step + 1)) {
            throw new IllegalArgumentException("Vector size does not match the specified range and step.");
        }
        for (int i = start; i <= end; i += step) {
            vector.set(i - start, i);
        }
    }

    /**
     * 向量的常数填充生成
     */
    public static void fillVector(double value, int length, Vector vector){
        if(vector.size() != length) {
            throw new IllegalArgumentException("Vector size does not match the specified length.");
        }
        for (int i = 0; i < length; i++) {
            vector.set(i, value);
        }
    }

    /** 
    * 向量的随机数填充 
    */
    public static void randomVector(double min, double max, Vector vector){
        if(vector.size() != vector.size()) {
            throw new IllegalArgumentException("Vector size does not match the specified length.");
        }
        for (int i = 0; i < vector.size(); i++) {
            double randomValue = min + Math.random() * (max - min);
            vector.set(i, randomValue);
        }
    }

    /**
     * 向量加法,无返回值，不覆盖
     */
    public static void add(Vector v1, Vector v2, Vector result) {
        if (v1.size() != v2.size()) {
            throw new IllegalArgumentException("Vectors must be of the same length for addition.");
        }
        if (result.size() != v1.size()) {
            throw new IllegalArgumentException("Result vector must be of the same length as the input vectors for addition.");
        }
        for (int i = 0; i < v1.size(); i++) {
            result.set(i, v1.get(i) + v2.get(i));
        }
    }

    /**
     * 向量加法，无返回值，覆盖
     */
    public static void add(Vector v1, Vector v2, boolean left){
        if (left) {
            for (int i = 0; i < v1.size(); i++) {
                v1.set(i, v1.get(i) + v2.get(i));
            }
        } else {
            for (int i = 0; i < v1.size(); i++) {
                v2.set(i, v1.get(i) + v2.get(i));
            }
        }
    }

    /**
     * 向量的标量加法，无返回值，不覆盖
     */
    public static void add(double scalar, Vector vector, Vector result) {
        if (result.size() != vector.size()) {
            throw new IllegalArgumentException("Result vector must be of the same length as the input vector for addition.");
        }
        for (int i = 0; i < vector.size(); i++) {
            result.set(i, vector.get(i) + scalar);
        }
    }
    /**
     * 向量的标量加法，无返回值，覆盖
     */
    public static void add(double scalar, Vector vector){
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, vector.get(i) + scalar);
        }
    }

    /**
     * 向量减法,无返回值，不覆盖
     */
    public static void subtract(Vector v1, Vector v2, Vector result) {
        if (v1.size() != v2.size()) {
            throw new IllegalArgumentException("Vectors must be of the same length for subtraction.");
        }
        if (result.size() != v1.size()) {
            throw new IllegalArgumentException("Result vector must be of the same length as the input vectors for subtraction.");
        }
        for (int i = 0; i < v1.size(); i++) {
            result.set(i, v1.get(i) - v2.get(i));
        }
    }

    /** 
     * 向量减法，无返回值，覆盖
    */
    public static void subtract(Vector v1, Vector v2, boolean left){
        if (left) {
            for (int i = 0; i < v1.size(); i++) {
                v1.set(i, v1.get(i) - v2.get(i));
            }
        } else {
            for (int i = 0; i < v1.size(); i++) {
                v2.set(i, v1.get(i) - v2.get(i));
            }
        }
    }

    /**
     * 向量的标量减法，无返回值，不覆盖
     */
    public static void subtract(double scalar, Vector vector, Vector result) {
        if (result.size() != vector.size()) {
            throw new IllegalArgumentException("Result vector must be of the same length as the input vector for subtraction.");
        }
        for (int i = 0; i < vector.size(); i++) {
            result.set(i, vector.get(i) - scalar);
        }
    }
    /**
     * 向量的标量减法，无返回值，覆盖
     */
    public static void subtract(double scalar, Vector vector){
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, vector.get(i) - scalar);
        }
    }

    /**
     * 向量数乘，无返回值，不覆盖
     */
    public static void multiply(double scalar, Vector vector, Vector result) {
        if (result.size() != vector.size()) {
            throw new IllegalArgumentException("Result vector must be of the same length as the input vector for multiplication.");
        }
        for (int i = 0; i < vector.size(); i++) {
            result.set(i, scalar * vector.get(i));
        }
    }

    /**
     * 向量数乘，无返回值，覆盖
     */
    public static void multiply(double scalar, Vector vector){
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, scalar * vector.get(i));
        }
    }

    /**
     * 向量点乘，有返回值
     */
    public static double dot(Vector v1, Vector v2) {
        if (v1.size() != v2.size()) {
            throw new IllegalArgumentException("Vectors must be of the same length for dot product.");
        }
        double result = 0.0;
        for (int i = 0; i < v1.size(); i++) {
            result += v1.get(i) * v2.get(i);
        }
        return result;
    }

    /**
     * 向量的模
     */
    public static double magnitude(Vector vector) {
        double sum = 0.0;
        for (int i = 0; i < vector.size(); i++) {
            sum += vector.get(i) * vector.get(i);
        }
        return Math.sqrt(sum);
    }

    /**
     * 向量的单位化，无返回值，不覆盖
     */
    public static void normalize(Vector vector, Vector result) {
        double magnitude = magnitude(vector);
        if (magnitude == 0.0) {
            throw new IllegalArgumentException("Cannot normalize a vector with zero magnitude.");
        }
        multiply(1.0 / magnitude, vector, result);
    }

    /**
     * 向量的单位化，无返回值，覆盖
     */
    public static void normalize(Vector vector){
        double magnitude = magnitude(vector);
        if (magnitude == 0.0) {
            throw new IllegalArgumentException("Cannot normalize a vector with zero magnitude.");
        }
        multiply(1.0 / magnitude, vector);
    }

    /**
     * 向量的夹角，有返回值
     */
    public static double angle(Vector v1, Vector v2) {
        double dotProduct = dot(v1, v2);
        double magnitudeProduct = magnitude(v1) * magnitude(v2);
        return Math.acos(dotProduct / magnitudeProduct);
    }

    /**
     * 向量的投影，无返回值，不覆盖
     */
    public static void project(Vector v1, Vector v2, Vector result) {
        double dotProduct = dot(v1, v2);
        double magnitudeSquared = magnitude(v2) * magnitude(v2);
        double scalar = dotProduct / magnitudeSquared;
        multiply(scalar, v2, result);
    }



}
