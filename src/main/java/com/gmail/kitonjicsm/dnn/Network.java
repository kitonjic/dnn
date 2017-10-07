package com.gmail.kitonjicsm.dnn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.la4j.Matrix;
import org.la4j.Vector;

public class Network {

    private int[] sizesOfLayers;
    private int numberOfLayers;
    private List<Vector> biases;
    private List<Matrix> weights;

    public Network(int[] sizesOfLayers) {
        this.sizesOfLayers = sizesOfLayers;
        this.numberOfLayers = sizesOfLayers.length;
        this.biases = initBiases();
        this.weights = initWeights();
    }

    public Vector feedForward(Vector a) {
        Vector output = a;
        for (int i = 0; i < numberOfLayers - 1; i++) {
            log(output, i);
            List<Double> values = new ArrayList<>();
            for (int column = 0; column < weights.get(i).columns(); column++) {
                Vector weight = weights.get(i).getColumn(column);
                values.add(output.innerProduct(weight));
            }
            output = Vector.fromCollection(values);
            output = output.add(biases.get(i));
            output = sigmoid(output);
        }
        log(output, numberOfLayers - 1);
        return output;
    }

    private Vector sigmoid(Vector vector) {
        List<Double> values = new ArrayList<>();
        for (int i = 0; i < vector.length(); i++) {
            values.add(1.0 / (1.0 + Math.exp(vector.get(i))));
        }
        return Vector.fromCollection(values);
    }

    private List<Vector> initBiases() {
        List<Vector> randomBiases = new ArrayList<>();
        for (int i = 1; i < numberOfLayers; i++) {
            int currentLayerSize = sizesOfLayers[i];
            randomBiases.add(Vector.random(currentLayerSize, new Random()));
        }
        return randomBiases;
    }

    private List<Matrix> initWeights() {
        ArrayList<Matrix> randomWeights = new ArrayList<>();
        for (int i = 0; i < numberOfLayers - 1; i++) {
            int currentLayerSize = sizesOfLayers[i];
            int nextLayerSize = sizesOfLayers[i + 1];
            randomWeights.add(Matrix.random(currentLayerSize, nextLayerSize, new Random()));
        }
        return randomWeights;
    }

    private void log(Vector output, int i) {
        System.out.println("LAYER: " + i);
        System.out.println(output);
        System.out.println();
    }
}
