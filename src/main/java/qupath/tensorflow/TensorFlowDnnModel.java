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

import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.bytedeco.opencv.opencv_core.Mat;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.Tensor;

import qupath.lib.io.UriResource;
import qupath.opencv.dnn.BlobFunction;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.dnn.DnnShape;
import qupath.opencv.dnn.PredictionFunction;


class TensorFlowDnnModel implements DnnModel<Tensor>, UriResource {
	
	private URI uri;
	
	private transient TensorFlowBundle bundle;

	private transient BlobFunction<Tensor> blobFunction;
	private transient PredictionFunction<Tensor> predictFunction;
	
	TensorFlowDnnModel(URI uri) {
		this.uri = uri;
	}
	
	private BlobFunction<Tensor> createBlobFunction() {
		return new TFBlobFun();
	}
	
	private PredictionFunction<Tensor> createPredictionFunction() {
		return new TFPredictionFunction();
	}

	private TensorFlowBundle getBundle() {
		if (bundle == null) {
			synchronized(this) {
				if (bundle == null)
					bundle = TensorFlowBundle.loadBundle(uri);
			}
		}
		return bundle;
	}

	@Override
	public BlobFunction<Tensor> getBlobFunction() {
		if (blobFunction == null) {
			synchronized(this) {
				if (blobFunction == null) {
					blobFunction = createBlobFunction();
				}			
			}
		}
		return blobFunction;
	}

	@Override
	public BlobFunction<Tensor> getBlobFunction(String name) {
		return getBlobFunction();
	}

	@Override
	public PredictionFunction<Tensor> getPredictionFunction() {
		if (predictFunction == null) {
			synchronized(this) {
				if (predictFunction == null) {
					predictFunction = createPredictionFunction();
				}			
			}
		}
		return predictFunction;
	}
	
	static class TFBlobFun implements BlobFunction<Tensor> {

		@Override
		public Tensor toBlob(Mat... mats) {
			return TensorFlowTools.convertToTensorFloat32(mats);
		}

		@Override
		public List<Mat> fromBlob(Tensor blob) {
			return TensorFlowTools.convertToMats(blob);
		}
		
	}
	
	ConcreteFunction getFunction() {
		return getBundle().getFunction();
	}
	
	
	class TFPredictionFunction implements PredictionFunction<Tensor> {
		
		@Override
		public Map<String, Tensor> predict(Map<String, Tensor> input) {
			var function = getFunction();
			return function.call(input);
		}


		@Override
		public Tensor predict(Tensor input) {
			var function = getFunction();
			return function.call(input);
		}

		@Override
		public Map<String, DnnShape> getInputs() {
			return getBundle().getInputs().stream().collect(Collectors.toMap(
					i -> i.getName(),
					i -> DnnShape.of(i.getShape())
					));
		}

		@Override
		public Map<String, DnnShape> getOutputs(DnnShape... inputShapes) {
			return getBundle().getOutputs().stream().collect(Collectors.toMap(
					i -> i.getName(),
					i -> DnnShape.of(i.getShape())
					));
		}
		
	}

	@Override
	public Collection<URI> getUris() throws IOException {
		return Collections.singletonList(uri);
	}

	@Override
	public boolean updateUris(Map<URI, URI> replacements) throws IOException {
		var replace = replacements.getOrDefault(uri, null);
		if (replace != null && !Objects.equals(uri, replace)) {
			uri = replace;
			return true;
		}
		return false;
	}

}
