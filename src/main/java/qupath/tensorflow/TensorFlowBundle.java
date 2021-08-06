/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2021 QuPath developers, The University of Edinburgh
 * %%
 * QuPath is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * QuPath is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License 
 * along with QuPath.  If not, see <https://www.gnu.org/licenses/>.
 * #L%
 */

package qupath.tensorflow;

import java.io.File;
import java.net.URI;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.TensorInfo;

class TensorFlowBundle {

	private final static Logger logger = LoggerFactory.getLogger(TensorFlowBundle.class);

	private String pathModel;
	private SavedModelBundle bundle;

	private String signatureDefKey;

	private List<SimpleTensorInfo> inputs;
	private List<SimpleTensorInfo> outputs;

	private MetaGraphDef metaGraphDef;

	private TensorFlowBundle(String pathModel) {

		this.pathModel = pathModel;

		var dir = new File(pathModel);
		if (!dir.exists()) {
			throw new IllegalArgumentException(pathModel + " does not exist!");
		} else if (!dir.isDirectory()) {
			throw new IllegalArgumentException(pathModel + " is not a valid TensorFlow model directory!");
		}


		// Load the bundle
		bundle = SavedModelBundle.load(pathModel, "serve");

		metaGraphDef = bundle.metaGraphDef();

		for (var entry : metaGraphDef.getSignatureDefMap().entrySet()) {
			var sigdef = entry.getValue();
			if (inputs == null || inputs.isEmpty()) {
				logger.info("Found SignatureDef: {} (method={})", entry.getKey(), sigdef.getMethodName());
				signatureDefKey = entry.getKey();
				inputs = sigdef.getInputsMap().values().stream().map(t -> new SimpleTensorInfo(t)).collect(Collectors.toList());
				outputs = sigdef.getOutputsMap().values().stream().map(t -> new SimpleTensorInfo(t)).collect(Collectors.toList());
			} else {
				logger.warn("Extra SignatureDef found - will be ignored ({}, method={})", entry.getKey(), sigdef.getMethodName());
			}
		}

		if (inputs.size() != 1) {
			logger.warn("Inputs: {}", inputs);
		}
		if (outputs.size() != 1) {
			logger.warn("Outputs: {}", outputs);
		}

		logger.info("Loaded {}", this);
	}
	
	
    private static Map<String, TensorFlowBundle> cachedBundles = new HashMap<>();

    static TensorFlowBundle loadBundle(String path) {
    	return cachedBundles.computeIfAbsent(path, p -> new TensorFlowBundle(p));
    }

    static TensorFlowBundle loadBundle(URI uri) {
    	return loadBundle(Paths.get(uri).toAbsolutePath().toString());
    }

    public ConcreteFunction getFunction() {
    	return bundle.function(signatureDefKey);
    }

	/**
	 * Get the path to the model (a directory).
	 * @return
	 */
	public String getModelPath() {
		return pathModel;
	}

	public long[] getOutputShape(String name) {
		var op = bundle.graph().operation(name);
		if (op == null)
			return null;
		int nOutputs = op.numOutputs();
		if (nOutputs > 1) {
			logger.warn("Operation {} has {} outputs!", name, nOutputs);
		} else if (nOutputs == 0)
			return new long[0];
		var shapeObject = op.output(0).shape();
		long[] shape = new long[shapeObject.numDimensions()];
		for (int i = 0; i < shape.length; i++)
			shape[i] = shapeObject.size(i);
		return shape;
	}

	/**
	 * Get information about the first required output (often the only one).
	 * @return
	 */
	public SimpleTensorInfo getInput() {
		return inputs == null || inputs.isEmpty() ? null : inputs.get(0);
	}

	/**
	 * Get information about all required inputs, or an empty list if no information is available.
	 * @return
	 */
	public List<SimpleTensorInfo> getInputs() {
		return inputs == null ? Collections.emptyList() : Collections.unmodifiableList(inputs);
	}

	/**
	 * Get the first provided output (often the only one).
	 * @return
	 */
	public SimpleTensorInfo getOutput() {
		return outputs == null || outputs.isEmpty() ? null : outputs.get(0);
	}

	/**
	 * Get information about all provided outputs, or an empty list if no information is available.
	 * @return
	 */
	public List<SimpleTensorInfo> getOutputs() {
		return outputs == null ? Collections.emptyList() : Collections.unmodifiableList(outputs);
	}

	/**
	 * Returns true if the model takes a single input.
	 * @return
	 */
	public boolean singleInput() {
		return inputs != null && inputs.size() == 1;
	}

	/**
	 * Returns true if the model provides a single output.
	 * @return
	 */
	public boolean singleOutput() {
		return outputs != null && outputs.size() == 1;
	}

	@Override
	public String toString() {
		if (singleInput() && singleOutput())
			return String.format("TensorFlow bundle: %s, (input=%s, output=%s)",
					getModelPath(), getInput(), getOutput());
		return String.format("TensorFlow bundle: %s, (inputs=%s, outputs=%s)",
				getModelPath(), getInputs(), getOutputs());
		//    	return String.format("TensorFlow bundle: %s, (input%s [%s], output=%s [%s])",
		//    			pathModel, inputName, arrayToString(inputShape), outputName, arrayToString(outputShape));
	}

	/**
	 * Helper class for parsing the essential info for an input/output tensor.
	 */
	public static class SimpleTensorInfo {

		private TensorInfo info;
		private String name;
		private long[] shape;

		SimpleTensorInfo(TensorInfo info) {
			this.info = info;
			this.name = info.getName();
			if (info.hasTensorShape()) {
				var dims = info.getTensorShape().getDimList();
				shape = new long[dims.size()];
				for (int i = 0; i < dims.size(); i++) {
					var d = dims.get(i);
					shape[i] = d.getSize();
				}
			}
		}

		TensorInfo getInfo() {
			return info;
		}

		/**
		 * Get any name associated with the tensor.
		 * @return
		 */
		public String getName() {
			return name;
		}

		/**
		 * Get the tensor shape as an array of long.
		 * @return
		 */
		public long[] getShape() {
			return shape == null ? null : shape.clone();
		}

		@Override
		public String toString() {
			if (shape == null) {
				return name + " (no shape)";
			} else {
				return name + " (" +
						LongStream.of(shape).mapToObj(l -> Long.toString(l)).collect(Collectors.joining(", "))
						+ ")";
			}
		}

	}


}
