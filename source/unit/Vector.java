package source.unit;

import java.util.ArrayList;
import java.util.List;

/**
 *  @HiCharm
 *  @Date: 2026-01-21 21:04
 *  @Description: 该向量类，仅作为数据存储单元
 */
public class Vector {
    public List<Double> elements = new ArrayList<>();

    /** 
     * 默认构造函数
     * 长度为0
     */
    public Vector() {   
    }

    /** 
     * 指定长度的构造函数
     */
    public Vector(int length) {
        for (int i = 0; i < length; i++) {
            elements.add(0.0);
        }
    }

    /**
     * 使用数组初始化的构造函数
     */
    public Vector(double[] array) {
        for (double element : array) {
            elements.add(element);
        }
    }

    /**
     * 拷贝构造函数
     */
    public Vector(Vector vector) {
        for (Double element : vector.elements) {
            this.elements.add(element);
        }
    }

    /**
     * 复制传入函数
     */
    public void copy(Vector vector) {
        this.elements.clear();
        for (Double element : vector.elements) {
            this.elements.add(element);
        }
    }

    /**
     * 复制传出函数
     */
    public Vector copy() {
        Vector newVector = new Vector();
        for (Double element : this.elements) {
            newVector.elements.add(element);
        }
        return newVector;
    }

    // 基础的getset方法

    public int size() {
        return elements.size();
    }

    public double get(int index) {
        return elements.get(index);
    }

    public void set(int index, double value) {
        elements.set(index, value);
    }

    /**
     * toString方法
     */
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < elements.size(); i++) {
            sb.append(elements.get(i));
            if (i < elements.size() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
