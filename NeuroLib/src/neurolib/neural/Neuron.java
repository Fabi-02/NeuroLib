package neurolib.neural;

import java.util.ArrayList;

import neurolib.utils.MathHelper;

/**
 * @author Fabian
 */

public class Neuron {

	protected static final int BIAS = 1;

	protected ArrayList<Double> inputs;
	protected ArrayList<Double> weights;
	protected double biasWeight;
	protected double output;

	protected double delta = 0;

	public Neuron() {
		this.inputs = new ArrayList<>();
		this.weights = new ArrayList<>();
	}

	public void setInputs(ArrayList<Double> inputs) {
		if (this.inputs.size() == 0 && weights.size() == 0) {
			this.inputs = new ArrayList<>(inputs);
			generateRandomWeights();
		}

		this.inputs = new ArrayList<>(inputs);
	}

	private void generateRandomWeights() {
		double range = 2 / (double) Math.sqrt(inputs.size());
		range = range > 0.6 ? range : 0.6;
		this.biasWeight = (Math.random() * range) - (range / 2);
		for (int i = 0; i < inputs.size(); i++) {
			weights.add((Math.random() * range) - (range / 2));
		}
	}

	private void calculateOutput() {
		double sum = 0;

		for (int i = 0; i < inputs.size(); i++) {
			sum += inputs.get(i) * weights.get(i);
		}
		sum += BIAS * biasWeight;

		output = sum;
	}

	public void adjustWeights(double delta, double learning_ratio) {
		this.delta = delta;
		for (int i = 0; i < inputs.size(); i++) {
			double d = weights.get(i);
			d -= learning_ratio * delta * inputs.get(i);
			weights.set(i, d);
		}

		biasWeight -= learning_ratio * delta * BIAS;
	}

	public double out() {
		calculateOutput();
		return MathHelper.sigmoidValue(output);
	}

	protected double net() {
		calculateOutput();
		return output;
	}
}