import java.io.*;
import java.util.HashMap;

public class inspect {

    public static void main(String[] args) {
        BufferedReader reader;
        BufferedWriter writer;
        double entropy = 0.0;
        double error = 0.0;
        HashMap<String, Integer> yValues= new HashMap<>();
        String input;
        String yValue;
        String[] attributes;
        double numberOfExamples = 0.0;
        int columnNumber =0;

        try {
            reader = new BufferedReader(new FileReader(args[0]));
            if((input = reader.readLine()) != null) {
                attributes = input.split(",");
                columnNumber = attributes.length-1;
            }
            while((input = reader.readLine()) != null) {
                attributes = input.split(",");
                yValue = attributes[columnNumber];
                yValues.merge(yValue, 1, (a,b) -> a+b);
                numberOfExamples++;
            }
            reader.close();
        } catch (NullPointerException e) {
            System.err.println("Null pointer error: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("IO Error: " + e.getMessage());
        }

        Integer majority=0;
        double numberWrong = 0.0;
        double numberRight = 0.0;
        String majorityClass="";

        for(HashMap.Entry<String, Integer> entry : yValues.entrySet()) {
            if(entry.getValue() > majority) {
                majority = entry.getValue();
                majorityClass=entry.getKey();
            }
        }

        for(HashMap.Entry<String, Integer> entry : yValues.entrySet()) {
            double coefficient = (double)(entry.getValue())/numberOfExamples;
            entropy-=(coefficient * Math.log(coefficient)/Math.log(2));
        }

        numberRight = yValues.get(majorityClass);
        numberWrong = numberOfExamples - numberRight;
        error = numberWrong / (numberOfExamples);


        try {
            writer = new BufferedWriter(new FileWriter(args[1]));
            String entropyLine = "entropy: " + entropy + '\n';
            String errorLine = "error: " + error + '\n';

            writer.append(entropyLine);
            writer.append(errorLine);
            writer.close();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}
