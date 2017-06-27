/**
 * Created by pateg on 20/03/2017.
 */


import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import java.io.*;
import java.util.*;
import java.util.List;
import static oracle.jrockit.jfr.events.Bits.intValue;


class CSVReader {
    ArrayList<arraylist> parseCSV(String pathto) {
        try {
            // Establishing the path
            Reader in = new FileReader(pathto);
            Iterable<csvrecord> records = CSVFormat.EXCEL.parse(in);
            // Create an instance and read in the lines
            BufferedReader br = new BufferedReader(new FileReader(pathto));
            String line = br.readLine();
            // Determine the number of variables by splitting at the commas
            Integer len = line.split(",").length;
            // Defining our matrix
            ArrayList<arraylist> colMatrix =  new ArrayList<>();
            // For loop that iterates over each line in the CSV
            for (CSVRecord record : records) {
                ArrayList<double> tempVector = new ArrayList<>();
                // Now we use the length to add each variable to the vector
                for (int i = 0; i < len; i++){
                    Double rec = Double.parseDouble(record.get(i));
                    tempVector.add(rec);
                }
                // And add this vector as a row to the matrix
                colMatrix.add(tempVector);
            }
            return colMatrix;
 
        } catch(IOException e){
            e.printStackTrace();
        }
            return null;
    }
}

class DataProcess {
    ArrayList<arraylist> splitSet(ArrayList<arraylist> dataset, double splitRatio) {
        // Getting the dataset size and trainSet size
        int dataSize = dataset.size();
        double trainSize = dataSize * splitRatio;
        // Training and test set variables
        ArrayList<arraylist> trainSet = new ArrayList<>();
        ArrayList<arraylist> testSet = dataset;
        Random rn = new Random(dataSize);
 
        while (trainSet.size() < trainSize) {
            // Create a new index
            Integer rand = rn.nextInt(testSet.size());
            // Retrieve vector that corresponds to
            ArrayList<Double> tempVector = testSet.get(rand);
            // Switch the vector from one set to another
            trainSet.add(tempVector);
            testSet.remove(intValue(rand));
        }
        // Returning sets in an arrayList
        ArrayList<arraylist> splitsets = new ArrayList<>(Arrays.asList(trainSet, testSet));
        return splitsets;
    }
 
    // Method for retrieving a single column from the matrix
    ArrayList<double> getCol(ArrayList<arraylist> dataset, int col) {
        ArrayList<double> colArray = new ArrayList<double>();
        for (int i = 0; i < dataset.size(); i++) {
            ArrayList row = dataset.get(i);
            Double column = (Double) row.get(col);
            colArray.add(column);
        }
        return colArray;
    }
}

class Predictions {
 
    Double densityFunc(double x, double mean, double stdDev) {
        // Probability density function used for determining probability
        double expTerm = Math.exp(-(Math.pow(x - mean, 2.0) / (2 * Math.pow(stdDev, 2.0))));
        return (1 / (Math.sqrt((2 * Math.PI)) * stdDev)) * expTerm;
    }
 
    HashMap<Integer, Double> classProbability(HashMap<Integer, ArrayList<arraylist>> summaries, ArrayList vector) {
        HashMap<Integer, Double> probability = new HashMap<>();
        // For each class split in the summaries, in this case 0 and 1
        for (Map.Entry<Integer, ArrayList<arraylist>> entry : summaries.entrySet()) {
            // Assign the class key to the new hashmap
            probability.put(entry.getKey(), 1.0);
            // Iterating through each random variable in the vector and then
            // the matching mean and SD for the variables.
            for (int i = 0; i < entry.getValue().size(); i++) {               // for each in the values
                // Get the mean and SD of the variable at column i
                Double mean = (Double) entry.getValue().get(i).get(0);
                Double stdDev = (Double) entry.getValue().get(i).get(1);
                // Get the random variable from the vector at index i
                double x = (Double) vector.get(i);
                // Probabilities are multiplied together
                double probval = probability.get(entry.getKey()) * densityFunc(x, mean, stdDev);
                probability.put(entry.getKey(), probval);
            }
        }
 
        return probability;
    }
 
    double decidePredict(HashMap<Integer, ArrayList<ArrayList>> summaries, ArrayList vector) {
        // Retrieves a new hashmap from classProbability method
        HashMap<Integer, Double> probability = classProbability(summaries, vector);
        // Sets up null values
        // These are over written as the summaries are determined
        double hiLabel = 99;
        double hiProb = -1;
        // Step through the returned hash with probability for each class
        for (Map.Entry<Integer, Double> entry : probability.entrySet()) {
            // Makes the decision which class label to assign to the vector
            // If the next probability is higher, it will replace the lower one
            if (entry.getValue() > hiProb) {
                hiProb = entry.getValue();
                hiLabel = entry.getKey();
            }
        }
        return hiLabel;
    }
 
    ArrayList goPredict(HashMap<Integer, ArrayList<arraylist>> summaries, ArrayList<arraylist> testSet) {
        ArrayList<double> finalPredictions = new ArrayList<>();
        // Loops through every vector in the testSet
        for (int i = 0; i < testSet.size(); i++) {
            // summaries and vector sent to predict and stored in final predictions
            double result = decidePredict(summaries, testSet.get(i));
            finalPredictions.add(result);
        }
        return finalPredictions;
    }
 
    Double accuracy (ArrayList<ArrayList> matrix, ArrayList<double> predictions){
        int correct = 0;
        int len = matrix.get(0).size();
        // The class labels are checked against the predictions
        for (int i =0; i < matrix.size(); i++){
            double var_a = (Double) matrix.get(i).get(len-1);
            double var_b = predictions.get(i);
            // Increase count for correct predictions
            if (var_a == var_b){
                correct = correct + 1;
            }
        }
        double msize = matrix.size();
        // Normalize to a percent
        double accuracy = correct/msize*100;
 
        return accuracy;
    }
}

class Statistics {
 
        ArrayList<double> individualStats(ArrayList<double> variable) {
            // Calculate mean and standard deviation
            double seMean = meanList(variable);
            double seSD = stdDev(variable);
            ArrayList<double> eachStats = new ArrayList<>(Arrays.asList(seMean, seSD));
            return eachStats;
        }
 
        HashMap classStats(ArrayList<arraylist> trainset) {
            // Instance of class
            DataProcess process = new DataProcess();
            // Hashmap used to store summaries
            HashMap<Integer, ArrayList<arraylist>> summaries = new HashMap<>();
            int len = trainset.get(1).size(); //??
            // Arrays used to store the 0 and 1 class statistics
            ArrayList<arraylist> ready0 = new ArrayList<>();
            ArrayList<arraylist> ready1 = new ArrayList<>();
            // Iterates across columns
            for (int i = 0; i < (len - 1); i++) {
                // Gets last column that is the class identifier
                List<Double> idCol = process.getCol(trainset, (len - 1));
                // For each vector iterate across variables
                List<double> valCol = process.getCol(trainset, i);
                // Lists for the two classes
                ArrayList<double> list1 = new ArrayList<>();
                ArrayList<double> list0 = new ArrayList<>();
                // Loop to separate into two classes
                for (int j = 0; j < idCol.size(); j++) {
                    // Splits out vectors based on their class ID
                    if (idCol.get(j) == 0) {
                        list0.add(valCol.get(j));
                    } else {
                        list1.add(valCol.get(j));
                    }
                }
                // Creates mean and SD for each variable after being split into classes
                ready0.add(individualStats(list0));
                ready1.add(individualStats(list1));
            }
            // Stores these in the hashmap for return
            summaries.put(0, ready0);
            summaries.put(1, ready1);
            return summaries;
        }
        // Used to sum a column
        double sumList(ArrayList<Double> a) {
            double sum = 0;
            for (int i = 0; i < a.size(); i++) {
                double in = a.get(i);
                sum = sum + in;
            }
            return sum;
        }
        // Takes the sum and returns the mean
        double meanList(ArrayList<Double> a) {
            double mean = sumList(a) / a.size();
            return mean;
        }
        // Takes the mean and calculates the standard deviation
        double stdDev(ArrayList a) {
            double mean = meanList(a);
            double num = 0;
            for (int i = 0; i < a.size(); i++) {
                double listVal = (double) a.get(i);
                num = num + Math.pow((listVal - mean), 2);
            }
            num = Math.sqrt(num / (a.size() - 1));
            return num;
        }
    }
	
	

public class Classifier {
    public static void main(String[] args) {
        // Creating instances of all of our classes
        CSVReader parse = new CSVReader();
        DataProcess process = new DataProcess();
        Predictions pred = new Predictions();
        Statistics stats = new Statistics();
        // This block imports the data and saves it into the matrix
        String s = "C:/Path_to/TwoNormDataset.csv";
        ArrayList<arraylist> matrix = parse.parseCSV(s);
        // Here we split the data into training and test set based on the split ratio
        ArrayList splitsets = process.splitSet(matrix, .7);
        // Retrieving the arrays using get
        ArrayList trainSet = (ArrayList)splitsets.get(0);
        ArrayList testSet = (ArrayList)splitsets.get(1);
        // Create summaries from the trainingSet for each of the variables
        HashMap summaries = stats.classStats(trainSet);
        // Finally send the summaries to make a prediction from the testSet
        ArrayList predictions = pred.goPredict(summaries, testSet);
        // Finally we return the accuracy of our prediction
        System.out.println("accuracy");
        System.out.println(pred.accuracy(testSet, predictions));
    }
}





