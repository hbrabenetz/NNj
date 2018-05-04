package com.niton.brab.nn;
import java.util.ArrayList;
import java.util.Collections;

public class NeuronalNetwork {

	public double[][] nod;
	private double[][][] wij;
	private double[][] err;
	public double[] tru;
	private double learnRate;

	private int NLay;
	private int TopMax;
	private ArrayList<Integer> top;

	public NeuronalNetwork(ArrayList<Integer> topologie, double LearnRate) {

		top = topologie;
		learnRate = LearnRate;

		System.out.println("Hello from neural Network");

		NLay = topologie.size();
		TopMax = Collections.max(topologie);
		System.out.println("TopMax = " + TopMax);

		wij = new double[NLay][TopMax + 1][TopMax + 1];
		nod = new double[NLay][TopMax + 1];
		err = new double[NLay][TopMax];
		tru = new double[TopMax];

		for (int nTop = 0; nTop < NLay; ++nTop)
			for (int topMaxi = 0; topMaxi < TopMax + 1; ++topMaxi)
				for (int topMaxj = 0; topMaxj < TopMax + 1; ++topMaxj)
					wij[nTop][topMaxi][topMaxj] = Math.random() * 2.0 - 1.0;

	}

	private int calc(boolean learn) {

		for (int nTop = 0; nTop < NLay - 1; ++nTop) {

			nod[nTop][top.get(nTop)] = 1.0; // the fictive d

			for (int nnodj = 0; nnodj < top.get(nTop + 1); ++nnodj) {
				nod[nTop + 1][nnodj] = 0.0;

				for (int nnodi = 0; nnodi < top.get(nTop) + 1; ++nnodi)
					nod[nTop + 1][nnodj] += nod[nTop][nnodi] * wij[nTop][nnodi][nnodj];

				// lets squash here
				nod[nTop + 1][nnodj] = 1.0 / (1.0 + Math.pow(2.718, -1.0 * nod[nTop + 1][nnodj]));

			}
		}

		if (learn == false)
			return 0;

		// output layer
		for (int nnod = 0; nnod < top.get(NLay - 1); ++nnod)
			err[NLay - 1][nnod] = (tru[nnod] - nod[NLay - 1][nnod]) * nod[NLay - 1][nnod] * (1.0 - nod[NLay - 1][nnod]);

		// next layers backwards

		for (int nLay = NLay - 1 - 1; nLay > 0; --nLay)
			for (int nnodi = 0; nnodi < top.get(nLay); ++nnodi) {
				err[nLay][nnodi] = 0.0;

				for (int nnodj = 0; nnodj < top.get(nLay + 1); ++nnodj)
					err[nLay][nnodi] += wij[nLay][nnodi][nnodj] * err[nLay + 1][nnodj];

				err[nLay][nnodi] *= nod[nLay][nnodi] * (1.0 - nod[nLay][nnodi]);

			}

		for (int nTop = 0; nTop < NLay - 1; ++nTop)
			for (int nnodi = 0; nnodi < top.get(nTop) + 1; ++nnodi)
				for (int nnodj = 0; nnodj < top.get(nTop + 1); ++nnodj)
					wij[nTop][nnodi][nnodj] += learnRate * nod[nTop][nnodi] * err[nTop + 1][nnodj];

		return 0;
	}
	
	public void train(Training train) {
		for(int i = 0;i<train.getInput().length;i++) {
			double[] in = train.getInput()[i];
			for (int j = 0; j < in.length; j++) {
				nod[0][j] = in[j];
			}
			double[] out = train.getOutput()[i];
			for(int j = 0;j<out.length;j++) {
				tru[j] = out[j]; 
			}
			calc(true);
		}
	}
	public void fire(double[] input) {
		for (int i = 0; i < input.length; i++) {
			nod[0][i] = input[i];
		}
		calc(false);
	}
	public double getError(Training train) {
		for(int i = 0;i<train.getInput().length;i++) {
			double[] in = train.getInput()[i];
			for (int j = 0; j < in.length; j++) {
				nod[0][j] = in[j];
			}
			double[] out = train.getOutput()[i];
			for(int j = 0;j<out.length;j++) {
				tru[j] = out[j]; 
			}
			calc(false);
		}
	}

}
