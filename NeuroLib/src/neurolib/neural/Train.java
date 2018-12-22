package neurolib.neural;

import java.util.ArrayList;

/**
 * @author Fabian
 */

public class Train {

	private Network network;
	private ArrayList<TrainingSet> trainingSets;

	public Train(Network network) {
		this.network = network;
		trainingSets = new ArrayList<>();
	}

	public void train(int count, double learning_ratio) {
		if (!trainingSets.isEmpty()) {
			for (int i = 0; i < count; i++) {
				train(learning_ratio);
			}
		}
	}

	private void train(double learning_ratio) {
		if (!trainingSets.isEmpty()) {
			int index = ((int) (Math.random() * trainingSets.size()));
			TrainingSet set = trainingSets.get(index);
			ArrayList<Double> inputs = set.getInputs();
			ArrayList<Double> goodOutputs = set.getGoodOutputs();
			network.run(inputs);
			network.adjustWages(goodOutputs, learning_ratio);
		}
	}

	public void clear() {
		trainingSets.clear();
	}

	public void addTrainingSet(TrainingSet newSet) {
		trainingSets.add(newSet);
	}
}
