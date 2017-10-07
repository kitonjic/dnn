package com.gmail.kitonjicsm;

import com.gmail.kitonjicsm.dnn.Network;
import org.la4j.vector.dense.BasicVector;

public class Main {

    public static void main(String[] args) {
        Network network = new Network(new int[]{3, 2, 3, 5, 10, 2});
        BasicVector inputVector = new BasicVector(new double[]{0, 1, 0});
        network.feedForward(inputVector);
    }

}
