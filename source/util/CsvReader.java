package source.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import source.unit.Matrix;

public class CsvReader {
    /** 
    * 读取csv文件内的数据集，解析表头，并构建输出矩阵返回。 
    */
    public static Matrix readCsv(String filename){
        // 读取csv文件
        List<String[]> lines = new ArrayList<>();  
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {  
            String line = "";  
            while ((line = br.readLine())!= null) {  
                String[] values = line.split(",");  
                lines.add(values);  
            }  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
        // 解析表头
        String[] headers = lines.get(0);  
        int width = headers.length;  
        int height = lines.size() - 1;  
        // 构建输出矩阵
        Matrix matrix = new Matrix(width, height);  
        for (int i = 1; i < lines.size(); i++) {  
            String[] values = lines.get(i);  
            for (int j = 0; j < width; j++) {  
                double value = Double.parseDouble(values[j]);  
                matrix.set(i - 1, j, value);  
            }  
        }  
        return matrix;  

    }
}
