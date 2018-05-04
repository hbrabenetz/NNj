package com.niton.brab.nn;

import java.util.ArrayList;
import java.util.Random;

public class Launcher {

	public static void main(String[] args) {
		ArrayList<Integer> topologie = new ArrayList<>();
		topologie.add(new Integer(2));
		topologie.add(new Integer(9));
		topologie.add(new Integer(5));
		topologie.add(new Integer(1));
		NeuronalNetwork nn = new NeuronalNetwork(topologie, 0.2);
		double[][] in = new double[1000][2];
		double[][] out = new double[1000][1];
		Random r = new Random();
		for (int i = 0; i < in.length; i++) {
			do {
				in[i][0] = r.nextDouble();
				in[i][1] = r.nextDouble();
				out[i][0] = in[i][0] * in[i][1];
			}while(out[i][0] >= 1.0);
		}
		Training tr = new Training(in, out);
		for (int i = 0; i < 10000; i++) {
			nn.train(tr);
			System.out.println("Train complete");
		}
		nn.fire(new double[]{0.4,0.5});
		System.out.println("0.4 * 0.5 = "+nn.nod[3][0]);
		nn.fire(new double[]{0.1,0.1});
		System.out.println("0.1 * 0.1 = "+nn.nod[3][0]);
		nn.fire(new double[]{0.7,0.5});
		System.out.println("0.7 * 0.5 = "+nn.nod[3][0]);
		nn.fire(new double[]{0.5,0.7});
		System.out.println("0.5 * 0.7 = "+nn.nod[3][0]);
		nn.fire(new double[]{0.9,0.9});
		System.out.println("0.9 * 0.9 = "+nn.nod[3][0]);



	}

}
