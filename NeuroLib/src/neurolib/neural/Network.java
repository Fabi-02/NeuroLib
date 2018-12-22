package neurolib.neural;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import neurolib.utils.MathHelper;

/**
 * @author Fabian
 */

public class Network {

	private int[] layerSize;
	private int layerCount;
	protected Neuron[][] neurons; // All Neurons from layer 1 - x (not the first (0) layer)

	public Network(int... size) {
		this.layerSize = size;
		this.layerCount = size.length;
		this.neurons = new Neuron[layerCount - 1][];
		createNeurons();
		initRun();
	}

	public int getLayerCount() {
		return layerCount;
	}

	public int[] getLayerSize() {
		return layerSize;
	}

	public static Network load(String path, String name) throws IOException {
		File file = new File(path, name + ".nlib");
		if (file.exists()) {
			BufferedReader br = new BufferedReader(new FileReader(file));

			ArrayList<Integer> struc = new ArrayList<>();
			ArrayList<String> neuronValues = new ArrayList<>();

			String line = "";
			while ((line = br.readLine()) != null) {
				if (struc.isEmpty()) {
					for (String s : line.split("-")) {
						int l = Integer.valueOf(s);
						struc.add(l);
					}
				} else {
					neuronValues.add(line);
				}
			}

			br.close();

			int[] structure = struc.stream().mapToInt(Integer::intValue).toArray();
			Network net = new Network(structure);

			for (int i = 0; i < neuronValues.size(); i++) {
				String layerNeuronValue = neuronValues.get(i);
				String[] neuronValue = layerNeuronValue.split("/");
				for (int n = 0; n < net.neurons[i].length; n++) {
					Neuron neuron = net.neurons[i][n];
					String values = neuronValue[n];
					String[] weightStrings = values.split(";");
					neuron.biasWeight = Double.valueOf(weightStrings[0]);
					ArrayList<Double> weights = new ArrayList<>();
					for (int w = 1; w < weightStrings.length; w++) {
						weights.add(Double.valueOf(weightStrings[w]));
					}
					neuron.weights = weights;
				}
			}
			return net;
		}
		return null;
	}

	public void save(String path, String name) throws IOException {
		save(path, name, -1);
	}

	public void save(String path, String name, int digits) throws IOException {
		File file = new File(path, name + ".nlib");
		if (!file.exists()) {
			file.createNewFile();
		}
		FileOutputStream out = new FileOutputStream(file);
		String netStructure = "";
		for (int i = 0; i < layerCount; i++) {
			netStructure += getNeuronCountInLayer(i);
			if (i < layerCount - 1) {
				netStructure += "-";
			}
		}
		netStructure += "\n";
		out.write(netStructure.getBytes());
		out.flush();
		for (int l = 0; l < neurons.length; l++) {
			String neuronValues = "";
			for (int i = 0; i < getNeuronCountInLayer(l + 1); i++) {
				Neuron neuron = neurons[l][i];
				neuronValues += digits < 0 ? neuron.biasWeight + ";"
						: MathHelper.round(neuron.biasWeight, digits) + ";";
				for (int w = 0; w < neuron.weights.size(); w++) {
					double weight = neuron.weights.get(w);
					neuronValues += digits < 0 ? weight : MathHelper.round(weight, digits);
					if (w < neuron.weights.size() - 1) {
						neuronValues += ";";
					}
				}
				if (i < getNeuronCountInLayer(l + 1) - 1) {
					neuronValues += "/";
				}
				out.write(neuronValues.getBytes());
				out.flush();
				neuronValues = "";
			}
			out.write("\n".getBytes());
			out.flush();
		}
		out.close();
	}

	private void createNeurons() {
		for (int l = 1; l < layerSize.length; l++) {
			int amount = layerSize[l];
			this.neurons[l - 1] = new Neuron[amount];
			for (int a = 0; a < amount; a++) {
				Neuron neuron = new Neuron();
				this.neurons[l - 1][a] = neuron;
			}
		}
	}

	public int getNeuronCountInLayer(int layer) {
		return layer < layerCount && layer >= 0 ? layerSize[layer] : -1;
	}

	private void initRun() {
		ArrayList<Double> input = new ArrayList<>();
		for (int i = 0; i < getNeuronCountInLayer(0); i++) {
			input.add(0.0);
		}
		run(input);
	}

	public ArrayList<Double> run(ArrayList<Double> input) {
		return run(input, 0);
	}

	@SuppressWarnings("unchecked")
	public ArrayList<Double> run(ArrayList<Double> input, int layer) {
		if (input.size() == getNeuronCountInLayer(layer)) {
			ArrayList<Double> outputs = new ArrayList<>();
			for (int i = layer; i < neurons.length; i++) {
				outputs.clear();
				for (Neuron neuron : neurons[i]) {
					neuron.setInputs(input);
					outputs.add(neuron.out());
				}
				input = (ArrayList<Double>) outputs.clone();
			}
			return outputs;
		}
		return null;
	}

	public ArrayList<Double> getOutputFromLayer(int layer) {
		if (layer > 0) {
			layer--;
			if (layer < layerCount) {
				ArrayList<Double> outputs = new ArrayList<>();
				for (Neuron neuron : neurons[layer]) {
					outputs.add(neuron.out());
				}
				return outputs;
			}
		} else if (layer == 0) {
			ArrayList<Double> outputs = new ArrayList<>(neurons[0][0].inputs);
			return outputs;
		}

		return null;
	}

	public ArrayList<ArrayList<Double>> getInputFromLayer(int layer) {
		if (layer > 0) {
			layer--;
			if (layer < layerCount) {
				ArrayList<ArrayList<Double>> inputs = new ArrayList<>();
				for (Neuron neuron : neurons[layer]) {
					inputs.add(neuron.inputs);
				}
				return inputs;
			}
		} else if (layer == 0) {
			ArrayList<Double> ins = new ArrayList<>(neurons[0][0].inputs);
			ArrayList<ArrayList<Double>> inputs = new ArrayList<>();
			for (Double input : ins) {
				ArrayList<Double> in = new ArrayList<>();
				in.add(input);
				inputs.add(in);
			}
			return inputs;
		}
		return null;
	}

	public ArrayList<ArrayList<Double>> getWeightsFromLayer(int layer) {
		if (layer > 0) {
			layer--;
			if (layer < layerCount) {
				ArrayList<ArrayList<Double>> weights = new ArrayList<>();
				for (Neuron neuron : neurons[layer]) {
					weights.add(neuron.weights);
				}
				return weights;
			}
		} else if (layer == 0) {
			ArrayList<ArrayList<Double>> inputs = new ArrayList<>();
			return inputs;
		}
		return null;
	}

	protected void adjustWages(ArrayList<Double> goodOutput, double learning_ratio) {

		// output neurons
		for (int i = 0; i < getNeuronCountInLayer(layerCount - 1); i++) {
			Neuron neuron = neurons[layerCount - 2][i];
			double neuronOut = neuron.out();
			double delta = (neuronOut * (1 - neuronOut)) * (neuronOut - goodOutput.get(i));
			neuron.adjustWeights(delta, learning_ratio);
		}

		// hidden neurons
		for (int n = layerCount - 3; n >= 0; n--) {
			for (int i = 0; i < getNeuronCountInLayer(n + 1); i++) {
				Neuron neuron = neurons[n][i];
				double neuronOut = neuron.out();
				double value = 0;
				for (int t = 0; t < getNeuronCountInLayer(n + 2); t++) {
					Neuron target = neurons[n + 1][t];
					value += target.delta * target.weights.get(i);
				}
				double delta = (neuronOut * (1 - neuronOut)) * value;
				neuron.adjustWeights(delta, learning_ratio);
			}
		}
	}
}
