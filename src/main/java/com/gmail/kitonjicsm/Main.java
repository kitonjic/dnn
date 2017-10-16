package com.gmail.kitonjicsm;

import static com.gmail.kitonjicsm.dnn.Network.NetworkBuilder.customNetwork;

import com.gmail.kitonjicsm.dnn.Network;
import org.la4j.vector.dense.BasicVector;

public class Main {

    public static void main(String[] args) {
        Network network = customNetwork()
                .learningRate(0.0001)
                .addLayerWithSize(3)
                .addLayerWithSize(2)
                .addLayerWithSize(3)
                .addLayerWithSize(5)
                .addLayerWithSize(2)
                .build();
        BasicVector inputVector = new BasicVector(new double[]{0, 1, 0});
        network.feedForward(inputVector);
    }

}
