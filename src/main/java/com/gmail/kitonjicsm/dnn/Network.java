package com.gmail.kitonjicsm.dnn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.la4j.Matrix;
import org.la4j.Vector;

public class Network {

    private static final double DEFAULT_LEARNING_RATE = 0.0001;

    private int[] sizesOfLayers;
    private int numberOfLayers;
    private List<Vector> biases;
    private List<Matrix> weights;
    private double learningRate = DEFAULT_LEARNING_RATE;

    private Network(NetworkBuilder builder) {
        this.sizesOfLayers = builder.sizesOfLayers;
        this.numberOfLayers = this.sizesOfLayers.length;
        this.learningRate = builder.learningRate;
        this.biases = initBiases();
        this.weights = initWeights();
    }

    public void train(Vector data, Vector expected) {
        // TODO stochastic gradient descent
    }

    public Vector feedForward(Vector a) {
        Vector output = a;
        for (int i = 0; i < numberOfLayers - 1; i++) {
            printData(output, i);
            List<Double> values = new ArrayList<>();
            for (int column = 0; column < weights.get(i).columns(); column++) {
                Vector weight = weights.get(i).getColumn(column);
                values.add(output.innerProduct(weight));
            }
            output = Vector.fromCollection(values);
            output = output.add(biases.get(i));
            output = sigmoid(output);
        }
        printData(output, numberOfLayers - 1);
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

    private void printData(Vector output, int i) {
        System.out.println("LAYER: " + i);
        System.out.println(output);
        System.out.println();
    }

    public static class NetworkBuilder {

        private int[] sizesOfLayers;
        private List<Integer> sizes = new ArrayList<>();
        private double learningRate = DEFAULT_LEARNING_RATE;

        public static NetworkBuilder customNetwork() {
            return new NetworkBuilder();
        }

        public NetworkBuilder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public NetworkBuilder addLayerWithSize(int layerSize) {
            this.sizes.add(layerSize);
            return this;
        }

        public Network build() {
            sizesOfLayers = new int[sizes.size()];
            for (int i = 0; i < sizes.size(); i++) {
                sizesOfLayers[i] = sizes.get(i);
            }
            return new Network(this);
        }
    }
}
