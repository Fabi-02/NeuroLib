package neurolib.utils;

/**
 * @author Fabian
 */

public class MathHelper {

	public static double sigmoidValue(double input) {
		double output = 1 / (1 + Math.pow(Math.E, -input));
		return output;
	}

	public static double round(double input, int count) {
		double output = input;
		output *= Math.pow(10, count);
		output = Math.round(output);
		output /= Math.pow(10, count);
		return output;
	}
}
