package com.niton.brab.nn;

public class Training {
	private double[][] input;
	private double[][] output;
	public Training(double[][] input, double[][] output) {
		this.input = input;
		this.output = output;
	}
	public double[][] getInput() {
		return input;
	}
	public double[][] getOutput() {
		return output;
	}
	
}
