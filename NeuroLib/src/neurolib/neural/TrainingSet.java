package neurolib.neural;

import java.util.ArrayList;

/**
 * @author Fabian
 */

public class TrainingSet {

	private ArrayList<Double> inputs;
	private ArrayList<Double> goodOutputs;

	public TrainingSet(ArrayList<Double> inputs, ArrayList<Double> goodOutputs) {
		this.inputs = inputs;
		this.goodOutputs = goodOutputs;
	}

	public ArrayList<Double> getInputs() {
		return inputs;
	}

	public ArrayList<Double> getGoodOutputs() {
		return goodOutputs;
	}
}
