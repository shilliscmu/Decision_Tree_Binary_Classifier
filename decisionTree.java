import java.io.*;
import java.util.*;

public class decisionTree  {

    public static void main(String[] args) {
        double trainingError =0.0;
        double testingError =0.0;
        ArrayList<String> trainingData = readCSV(args[0]);
        ArrayList<String> testingData = readCSV(args[1]);
        int maxDepth = Integer.parseInt(args[2]);
        // train decision tree from trainingInput
        Node decisionTree = trainDecisionTree(trainingData, maxDepth);
        ArrayList<String> trainingPredictedLabels = generatePredictedLabels(decisionTree, trainingData);
        printLabels(trainingPredictedLabels, args[3]);
        trainingError = calculateLabelingError(trainingPredictedLabels, trainingData);
        // test decision tree on testingInput
        ArrayList<String> testingPredictedLabels = generatePredictedLabels(decisionTree, testingData);
        printLabels(testingPredictedLabels, args[4]);
        testingError = calculateLabelingError(testingPredictedLabels, testingData);
        // print metrics
        printMetrics(args[5], trainingError, testingError);
    }

    private static ArrayList<String> readCSV(String fileName) {
        BufferedReader reader;
        String input;
        ArrayList<String> csv = new ArrayList<>();
        try {
            reader = new BufferedReader(new FileReader(fileName));
            while((input = reader.readLine()) != null) {
                csv.add(input);
            }
            reader.close();
        } catch (NullPointerException e) {
            System.err.println("Null pointer error: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("IO Error: " + e.getMessage());
        }
        return csv;
    }

    private static Node trainDecisionTree(ArrayList<String> trainingData, int maxDepth) {
        int currentDepth=0;
        ArrayList<String> pastAttributes = new ArrayList<>();
        Node decisionTree = new Node(trainingData, currentDepth, pastAttributes);
        if(maxDepth == currentDepth) {
            decisionTree.setMajorityVote(majorityVoteLabel(decisionTree.getCSV()));
            return decisionTree;
        } else {
            TreeMap<String, Integer> yValues = new TreeMap<>();
            String[] attributes;
            for(int i = 1; i < trainingData.size(); i++) {
                attributes = trainingData.get(i).split(",");
                yValues.merge(attributes[attributes.length-1], 1, (a,b) -> a+b);
            }
            Set<String> temp = new HashSet<>(Arrays.asList(yValues.firstKey(), yValues.lastKey()));
            final ArrayList<String> twoYValues = new ArrayList<>(temp);
            System.out.println("[" + yValues.firstEntry().getValue() + " " + yValues.firstKey() + "/" + yValues.lastEntry().getValue() + " " + yValues.lastKey() + "]");
            decisionTree = trainStumpTree(trainingData, currentDepth, maxDepth, pastAttributes, twoYValues);
        }

        return decisionTree;
    }

    private static void printLabels(ArrayList<String> labels, String fileName) {
        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(fileName));
            for(String s : labels) {
                s+='\n';
                writer.append(s);
            }
            writer.close();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    private static void printMetrics(String fileName, double trainingError, double testingError) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
            String trainingErrorLine = "error(train): " + trainingError + '\n';
            String testingErrorLine = "error(test): " + testingError;

            writer.append(trainingErrorLine);
            writer.append(testingErrorLine);
            writer.close();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    //calculate entropy given ArrayList<String> csvTrainingData
    private static double calculateEntropy(ArrayList<String> csvTrainingData, int indexOfAttributeColumn) {
        String[] attributes={""};
        int numberOfExamples=0;
        double entropy = 0.0;
        HashMap<String, Integer> attributeValues = new HashMap<>();
        for(int i = 1; i < csvTrainingData.size(); i++) {
            attributes = csvTrainingData.get(i).split(",");
            attributeValues.merge(attributes[indexOfAttributeColumn], 1, (a,b) -> a+b);
            numberOfExamples++;
        }

        for(HashMap.Entry<String, Integer> entry : attributeValues.entrySet()) {
            double coefficient = (double)(entry.getValue())/numberOfExamples;
            entropy-=(coefficient * Math.log(coefficient)/Math.log(2));
        }
        return entropy;
    }

    private static double calculateJointEntropy(ArrayList<String> csvTrainingData, int indexOfAttributeX, int indexOfAttributeY) {
        String[] attributes={""};
        int numberOfExamples=0;
        double entropy = 0.0;
        Pair<String, String> jointAttributes;
        HashMap<Pair<String, String>, Integer> attributeValues = new HashMap<>();
        for(int i = 1; i < csvTrainingData.size(); i++) {
            attributes = csvTrainingData.get(i).split(",");
            jointAttributes = new Pair(attributes[indexOfAttributeX], attributes[indexOfAttributeY]);
            attributeValues.merge(jointAttributes, 1, (a,b) -> a+b);
            numberOfExamples++;
        }

        for(HashMap.Entry<Pair<String, String>, Integer> entry : attributeValues.entrySet()) {
            double coefficient = (double)(entry.getValue())/numberOfExamples;
            entropy-=(coefficient * Math.log(coefficient)/Math.log(2));
        }
        return entropy;
    }


    //calculate mutual information: entropy(Y) + entropy(X) - entropy(Y,X)
    private static double calculateMutualInformation(ArrayList<String> csvTrainingData, int indexOfAttributeX, int indexOfAttributeY) {
        return (calculateEntropy(csvTrainingData, indexOfAttributeY)
                + calculateEntropy(csvTrainingData, indexOfAttributeX)
                - calculateJointEntropy(csvTrainingData, indexOfAttributeX, indexOfAttributeY));
    }

    private static Node trainStumpTree(ArrayList<String> trainingData, int currentDepth, int maxDepth, ArrayList<String> pastAttributes, ArrayList<String> twoYValues) {
        Node stumpTree = new Node(trainingData, currentDepth, pastAttributes);
        ArrayList<String> splitAttributes = new ArrayList<>();
        ArrayList<String> leftNodeData = new ArrayList<>(Arrays.asList(trainingData.get(0)));
        ArrayList<String> rightNodeData = new ArrayList<>(Arrays.asList(trainingData.get(0)));
        Set<String> valuesOfAttributeToSplitOn = new HashSet<>();
        String[] attributes = trainingData.get(0).split(",");
        Collections.addAll(splitAttributes, Arrays.copyOfRange(attributes, 0, attributes.length-1));
        HashMap<String, Integer> unusedAttributes = new HashMap<>();
        double mutualInfo = 0.0;
        String attributeToSplitOn = "";

        stumpTree.setMajorityVote(majorityVoteLabel(stumpTree.getCSV()));

        for(String s : splitAttributes) {
            if((stumpTree.getPastAttributes() == null) || (stumpTree.getPastAttributes().isEmpty()) || (!stumpTree.getPastAttributes().contains(s))) {
                unusedAttributes.put(s, splitAttributes.indexOf(s));
            }
        }

//        System.out.println("Unused attributes and their indicies: " + unusedAttributes);

        if(unusedAttributes.isEmpty()) {
            return stumpTree;
        }
        if(!(currentDepth < maxDepth)) {
            return stumpTree;
        }
        for(HashMap.Entry<String, Integer> entry : unusedAttributes.entrySet()) {
            double temp = calculateMutualInformation(trainingData, entry.getValue(), attributes.length-1);
            if (temp > mutualInfo) {
                mutualInfo=temp;
                attributeToSplitOn = entry.getKey();
            }
        }

//        System.out.println("Attribute to split on is " + attributeToSplitOn + "with a mutual info of " + mutualInfo);

        if(attributeToSplitOn.isEmpty()) {
            return stumpTree;
        } else {
            stumpTree.setSplitAttribute(attributeToSplitOn);
            int columnNumber = splitAttributes.indexOf(attributeToSplitOn);
            TreeMap<String, Integer> yValuesForAttributeLeftValue = new TreeMap<>();
            TreeMap<String, Integer> yValuesForAttributeRightValue = new TreeMap<>();
            for (int i = 1; i < trainingData.size(); i++) {
                valuesOfAttributeToSplitOn.add(trainingData.get(i).split(",")[columnNumber]);
            }
            ArrayList<String> twoValuesOfAttributesToSplitOn = new ArrayList<>(valuesOfAttributeToSplitOn);
            String line;
            for (int i = 1; i < trainingData.size(); i++) {
                line = trainingData.get(i);
                if(twoValuesOfAttributesToSplitOn.get(0).equals(line.split(",")[columnNumber])) {
                    leftNodeData.add(line);
                    yValuesForAttributeLeftValue.merge(line.split(",")[attributes.length-1], 1, Integer::sum);
                } else {
                    rightNodeData.add(line);
                    yValuesForAttributeRightValue.merge(line.split(",")[attributes.length-1], 1, (a, b) -> a+b);
                }
            }

            String pp = "";
            for(int i = 0; i < stumpTree.getNodeDepth()+1; i++) {
                pp+="| ";
            }

            if(yValuesForAttributeLeftValue.size()==1) {
                if(twoYValues.get(0).equals(yValuesForAttributeLeftValue.firstKey())) {
                    yValuesForAttributeLeftValue.put(twoYValues.get(1), 0);
                } else {
                    yValuesForAttributeLeftValue.put(twoYValues.get(0), 0);
                }
            }
            if(yValuesForAttributeRightValue.size()==1) {
                if(twoYValues.get(0).equals(yValuesForAttributeRightValue.firstKey())) {
                    yValuesForAttributeRightValue.put(twoYValues.get(1), 0);
                } else {
                    yValuesForAttributeRightValue.put(twoYValues.get(0), 0);
                }
            }

            ArrayList<String> pastAttributesForChildren = new ArrayList<>();
            pastAttributesForChildren.addAll(stumpTree.getPastAttributes());
            pastAttributesForChildren.add(stumpTree.getSplitAttribute());

//            System.out.println("The function level past attributes are: " + pastAttributes);
//            System.out.println("The node's past attributes are" + stumpTree.getPastAttributes());

            String ppl = pp + prettyPrint(attributeToSplitOn, twoValuesOfAttributesToSplitOn.get(0), yValuesForAttributeLeftValue.firstKey(), yValuesForAttributeLeftValue.firstEntry().getValue(), yValuesForAttributeLeftValue.lastKey(), yValuesForAttributeLeftValue.lastEntry().getValue());
            System.out.println(ppl);
            stumpTree.setLeftNodeValue(twoValuesOfAttributesToSplitOn.get(0));
            stumpTree.setLeftNode(trainStumpTree(leftNodeData, stumpTree.getNodeDepth()+1, maxDepth, pastAttributesForChildren, twoYValues));

            String ppr = pp + prettyPrint(attributeToSplitOn, twoValuesOfAttributesToSplitOn.get(1), yValuesForAttributeRightValue.firstKey(), yValuesForAttributeRightValue.firstEntry().getValue(), yValuesForAttributeRightValue.lastKey(), yValuesForAttributeRightValue.lastEntry().getValue());
            System.out.println(ppr);
            stumpTree.setRightNodeValue(twoValuesOfAttributesToSplitOn.get(1));
            stumpTree.setRightNode(trainStumpTree(rightNodeData, stumpTree.getNodeDepth()+1, maxDepth, pastAttributesForChildren, twoYValues));
        }
        return stumpTree;
    }

    private static String majorityVoteLabel(ArrayList<String> data) {
        HashMap<String, Integer> yValues = new HashMap<>();
        String[] attributes;
        int columnNumber = 0;
        Integer majority =0;
        String majorityClass = "";
        if((data.get(0)!=null)) {
            attributes = data.get(0).split(",");
            columnNumber = attributes.length-1;
        }
        for (int i = 1; i < data.size(); i++) {
            yValues.merge(data.get(i).split(",")[columnNumber], 1, (a,b) -> a+b);
        }
        for(HashMap.Entry<String, Integer> entry : yValues.entrySet()) {
            if(entry.getValue() > majority) {
                majority = entry.getValue();
                majorityClass=entry.getKey();
            }
        }
        return majorityClass;
    }

    //take learned tree with data; generate predicted labels
    private static ArrayList<String> generatePredictedLabels(Node learnedDecisionTree, ArrayList<String> data) {
        ArrayList<String> predictedLabels = new ArrayList<>();
        ArrayList<String> attributes = new ArrayList<>();
        ArrayList<String> singleLineValues = new ArrayList<>();
        //has two strings: attribute line and single line instance
        ArrayList<String> singleLineData = new ArrayList<>(Arrays.asList(data.get(0), null));
        String splitAttribute = "";
        String splitAttributeValue = "";
        int indexOfSplitAttribute;
        //node with zero children
//        if((learnedDecisionTree.getLeftNodeValue()==null)&&(learnedDecisionTree.getRightNodeValue()==null)) {
//            String majorityVote = learnedDecisionTree.getMajorityVote();
//            //for loop assumes attribute row is present
//            for(int i=0; i < data.size()-1; i++) {
//                predictedLabels.add(majorityVote);
//            }
//        } else {
            splitAttribute = learnedDecisionTree.getSplitAttribute();
            Collections.addAll(attributes, data.get(0).split(","));
            indexOfSplitAttribute = attributes.indexOf(splitAttribute);
            String line;
            for(int i = 1; i < data.size(); i++) {
                line = data.get(i);
                singleLineData.set(1, line);
//                System.out.print("For instance " + i);
                predictedLabels.add(generatePredictedLabel(singleLineData, learnedDecisionTree));
//                Collections.addAll(singleLineValues, line.split(","));
//                if((splitAttribute!=null) && (!splitAttribute.isEmpty()) && (singleLineValues.size()>indexOfSplitAttribute)) {
//                    splitAttributeValue = singleLineValues.get(indexOfSplitAttribute);
//                    if((learnedDecisionTree.getLeftNodeValue() != null) && (!splitAttributeValue.isEmpty()) && (splitAttributeValue.equals(learnedDecisionTree.getLeftNodeValue()))) {
//                        predictedLabels.addAll(generatePredictedLabels(learnedDecisionTree.getLeftNode(), singleLineData));
//                        System.out.println("For line " + i + " we predicted the label " + )
//                    } if ((learnedDecisionTree.getRightNodeValue() != null) && (!splitAttributeValue.isEmpty()) &&(splitAttributeValue.equals(learnedDecisionTree.getRightNodeValue()))) {
//                        predictedLabels.addAll(generatePredictedLabels(learnedDecisionTree.getRightNode(), singleLineData));
//                    }
//                }
//                else {
//                    predictedLabels.add(learnedDecisionTree.getMajorityVote());
//                }
            }
//        }
        return predictedLabels;
    }

    private static String generatePredictedLabel(ArrayList<String> singleLineData, Node learnedDecisionTree) {
        String splitAttribute = learnedDecisionTree.getSplitAttribute();
        String splitAttributeValue = "";
        int indexOfSplitAttribute;
        ArrayList<String> singleLineValues = new ArrayList<>();
        Collections.addAll(singleLineValues, singleLineData.get(1).split(","));
        ArrayList<String> attributes = new ArrayList<>();
        Collections.addAll(attributes, singleLineData.get(0).split(","));
        indexOfSplitAttribute = attributes.indexOf(splitAttribute);
        String label = "";
        if((splitAttribute!=null) && (!splitAttribute.isEmpty()) && (singleLineValues.size()>indexOfSplitAttribute)) {
            splitAttributeValue = singleLineValues.get(indexOfSplitAttribute);
            if((learnedDecisionTree.getLeftNodeValue() != null) && (!splitAttributeValue.isEmpty()) && (splitAttribute.equals(learnedDecisionTree.getSplitAttribute())) && (splitAttributeValue.equals(learnedDecisionTree.getLeftNodeValue()))) {
                label = (generatePredictedLabel(singleLineData, learnedDecisionTree.getLeftNode()));
//                System.out.println(", we predicted the label " + label);
                return label;
            } if ((learnedDecisionTree.getRightNodeValue() != null) && (!splitAttributeValue.isEmpty()) && (splitAttribute.equals(learnedDecisionTree.getSplitAttribute())) && (splitAttributeValue.equals(learnedDecisionTree.getRightNodeValue()))) {
                label = (generatePredictedLabel(singleLineData, learnedDecisionTree.getRightNode()));
//                System.out.println(", we predicted the label " + label);
                return label;
            }
        }
        else {
            label = (learnedDecisionTree.getMajorityVote());
            return label;
        }
        return "bad things?";
    }

    //calculate error of predicted labels to true labels
    private static double calculateLabelingError(ArrayList<String> predictedLabels, ArrayList<String> data) {
        double error = 0.0;
        double size;
        ArrayList<String> trueLabels = new ArrayList<>();
        String[] attributes;

        for(int i = 1; i < data.size(); i++){
            attributes = data.get(i).split(",");
            trueLabels.add(attributes[attributes.length-1]);
        }

        if(trueLabels.equals(predictedLabels)) {
            return 0;
        }

        size = (predictedLabels.size() >= trueLabels.size()) ? predictedLabels.size() : trueLabels.size();
//        System.out.println(size);
        for(int i =0; i < size; i++) {
            if(!predictedLabels.get(i).equals(trueLabels.get(i))) {
                error+=1;
            }
        }

        return error/size;
    }

    private static String prettyPrint(String splitAttribute, String splitAttributeValue,
                                    String yValue1, int yValueCount1, String yValue2, int yValueCount2) {
        return(splitAttribute + " = " + splitAttributeValue + ": [" + yValueCount1 + " "
                      + yValue1 + " / " + yValueCount2 + " " + yValue2 + "]");
    }

    public static class Pair<F, S> extends java.util.AbstractMap.SimpleImmutableEntry<F, S> {
        public Pair(F f, S s) {
            super(f, s);
        }
    }

    public static class Node {

        public Node(ArrayList<String> trainingData, int currentDepth, ArrayList<String> pastAttributes){
            this.CSV = trainingData;
            this.nodeDepth = currentDepth;
            this.pastAttributes = pastAttributes;
            this.leftNode = null;
            this.leftNodeValue = null;
            this.rightNode = null;
            this.rightNodeValue = null;
            this.majorityVote = null;
        }

        private String splitAttribute;
        private ArrayList<String> pastAttributes;
        private int nodeDepth;
        private Node leftNode;
        private String leftNodeValue;
        private Node rightNode;
        private String rightNodeValue;
        private ArrayList<String> CSV;
        private String majorityVote;

        public String getSplitAttribute() {
            return splitAttribute;
        }

        public void setSplitAttribute(String splitAttribute) {
            this.splitAttribute = splitAttribute;
        }

        public ArrayList<String> getPastAttributes() {
            return pastAttributes;
        }

        public void setPastAttributes(ArrayList<String> pastAttributes) {
            this.pastAttributes = pastAttributes;
        }

        public int getNodeDepth() {
            return nodeDepth;
        }

        public void setNodeDepth(int nodeDepth) {
            this.nodeDepth = nodeDepth;
        }

        public Node getLeftNode() {
            return leftNode;
        }

        public void setLeftNode(Node leftNode) {
            this.leftNode = leftNode;
        }

        public Node getRightNode() {
            return rightNode;
        }

        public void setRightNode(Node rightNode) {
            this.rightNode = rightNode;
        }

        public String getLeftNodeValue() {
            return leftNodeValue;
        }

        public void setLeftNodeValue(String leftNodeValue) {
            this.leftNodeValue = leftNodeValue;
        }

        public String getRightNodeValue() {
            return rightNodeValue;
        }

        public void setRightNodeValue(String rightNodeValue) {
            this.rightNodeValue = rightNodeValue;
        }

        public ArrayList<String> getCSV() {
            return CSV;
        }

        public void setCSV(ArrayList<String> CSV) {
            this.CSV = CSV;
        }

        public String getMajorityVote() {
            return majorityVote;
        }

        public void setMajorityVote(String majorityVote) {
            this.majorityVote = majorityVote;
        }

    }
}
