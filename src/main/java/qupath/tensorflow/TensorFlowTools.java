/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2020 - 2021 QuPath developers, The University of Edinburgh
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
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.index.Index;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TUint8;
import qupath.lib.common.GeneralTools;
import qupath.lib.regions.Padding;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.tools.OpenCVTools;

/**
 * Helper methods for working with TensorFlow and QuPath, with the help of OpenCV.
 * 
 * @author Pete Bankhead
 */
public class TensorFlowTools {
	
	private final static Logger logger = LoggerFactory.getLogger(TensorFlowTools.class);
	
	static {
		logger.info("TensorFlow version {}", TensorFlow.version());
	}


	/**
	 * Convert a single {@link Mat} to a {@link Tensor}.
	 * <p>
	 * This supports only a subset of Mats according to type, namely
	 * <ul>
	 *   <li>CV_8U</li>
	 *   <li>CV_32S</li>
	 *   <li>CV_32F</li>
	 *   <li>CV_64F</li>
	 * </ul>
	 * 
	 * The input is assumed to be a 'standard' image (rows, columns, channels); output will have 1 pre-pended as a batch.
	 * This behavior may change in the future!
	 * <p>
	 *
	 * @param mat the input {@link Mat}
	 * @return the converted {@link Tensor}
	 * 
	 * @throws IllegalArgumentException if the depth is not supported
	 * @implNote the batch size of the output tensor is 1.
	 * @see #convertToTensorFloat32(Mat...)
	 */
	public static <T> Tensor convertToTensor(Mat mat) throws IllegalArgumentException {
	    int w = mat.cols();
	    int h = mat.rows();
	    int nBands = mat.channels();
	    var shape = Shape.of(1, h, w, nBands);
	    
	    mat = OpenCVTools.ensureContinuous(mat, false);
	    
	    int depth = mat.depth();
	    if (depth == opencv_core.CV_32F) {
	    	FloatBuffer buffer = mat.createBuffer();
	    	return TFloat32.tensorOf(shape, DataBuffers.of(buffer));
	    }
	    if (depth == opencv_core.CV_64F) {
	    	DoubleBuffer buffer = mat.createBuffer();
	    	return TFloat64.tensorOf(shape, DataBuffers.of(buffer));
	    }
	    if (depth == opencv_core.CV_32S) {
	    	IntBuffer buffer = mat.createBuffer();
	    	return TInt32.tensorOf(shape, DataBuffers.of(buffer));
	    }
	    if (depth == opencv_core.CV_8U) {
	    	ByteBuffer buffer = mat.createBuffer();
	    	return TUint8.tensorOf(shape, DataBuffers.of(buffer));
	    }
	    throw new IllegalArgumentException("Unsupported Mat depth! Must be 8U, 32S, 32F or 64F.");
	}
	
	
	/**
	 * Convert one or more Mats to a single 32-bit float tensor.
	 * @param mats
	 * @return
	 * @throws IllegalArgumentException
	 */
	public static TFloat32 convertToTensorFloat32(Mat... mats) throws IllegalArgumentException {
		
		if (mats.length == 0)
			throw new IllegalArgumentException("No Mats supplied to convertToTensor!");
				
		// Get dimensions
		// Here, we use NHWC (*not* the conventional OpenCV/QuPath ordering!)
		long[] shape = new long[] {mats.length, -1, -1, -1};
		for (var mat : mats) {
			if (shape[1] == -1) {
				shape[1] = mat.rows();
				shape[2] = mat.cols();
				shape[3] = mat.channels();
			} else if (shape[1] != mat.rows() || shape[2] != mat.cols() || shape[3] != mat.channels()) {
				logger.error("Incompatible shapes {} and {} found!", Arrays.toString(shape), Arrays.toString(mat.createIndexer().sizes()));
				throw new IllegalArgumentException("Mats provided to convertToTensor have incompatible shapes!");
			}
		}

		// Create an NdArray to store the data
		var data = NdArrays.ofFloats(Shape.of(shape));
    	
		long ind = 0;
		Index[] inds = Stream.of(shape).map(s -> Indices.all()).toArray(Index[]::new);
	    for (var mat : mats) {
	    	
	    	inds[0] = Indices.at(ind);
	    	var slice = data.slice(inds);
	    	
	    	var buffer = DataBuffers.of(getFloatBuffer(mat));
	    	slice.write(buffer);
	    		    	
	    	ind++;
	    }
		
		return TFloat32.tensorOf(data);
	}
	
	
	private static FloatBuffer getFloatBuffer(Mat mat) {
		var buffer = mat.createBuffer();
		if (buffer instanceof FloatBuffer)
			return (FloatBuffer)buffer;
		var mat2 = new Mat();
		mat.convertTo(mat2, opencv_core.CV_32F);
		return mat2.createBuffer();
	}
	

	/**
	 * Convert a {@link Tensor} to a {@link Mat}.
	 * Currently this is rather limited in scope:
	 * <ul>
	 *   <li>both input and output must be 32-bit floating point</li>
	 *   <li>the output is expected to be a 'standard' image (rows, columns, channels)</li>
	 *   <li>it <b>only</b> handles a batch size of 1, and the batch (first) dimension is dropped</li>
	 * </ul>
	 * This method may be replaced by something more customizable in the future. 
	 * 
	 * @param tensor
	 * @return
	 */
	@Deprecated
	static Mat convertToMat(Tensor tensor) {
		long[] shape = tensor.shape().asArray();
	    // Get the shape, stripping off the batch
	    int n = shape.length;
	    int[] dims = new int[Math.max(3, n-1)];
//	    int[] dims = new int[n-1];
	    Arrays.fill(dims, 1);
	    for (int i = 1; i < n; i++) {
	    	dims[i-1] = (int)shape[i];
	    }
	    // Get total number of elements (pixels)
	    int size = 1;
	    for (long d : dims)
	    	size *= d;
	    Mat mat = null;
	    switch (tensor.dataType()) {
		case DT_BFLOAT16:
			break;
		case DT_BOOL:
			break;
		case DT_COMPLEX128:
			break;
		case DT_COMPLEX64:
			break;
		case DT_DOUBLE:
		    mat = new Mat(dims[0], dims[1], opencv_core.CV_64FC(dims[2]));
		    DoubleBuffer buffer64F = mat.createBuffer();
		    if (buffer64F.hasArray())
			    tensor.asRawTensor().data().asDoubles().read(buffer64F.array());
		    else {
			    double[] values = new double[size];
			    tensor.asRawTensor().data().asDoubles().read(values);
			    buffer64F.put(values);
		    }
		    return mat;
		case DT_FLOAT:
		    mat = new Mat(dims[0], dims[1], opencv_core.CV_32FC(dims[2]));
		    FloatBuffer buffer32F = mat.createBuffer();
		    if (buffer32F.hasArray())
			    tensor.asRawTensor().data().asFloats().read(buffer32F.array());
		    else {
			    float[] values = new float[size];
			    tensor.asRawTensor().data().asFloats().read(values);
			    buffer32F.put(values);
		    }
		    return mat;
		case DT_HALF:
			break;
		case DT_INT16:
			break;
		case DT_INT32:
			mat = new Mat(dims[0], dims[1], opencv_core.CV_32SC(dims[2]));
		    IntBuffer buffer32S = mat.createBuffer();
		    if (buffer32S.hasArray())
			    tensor.asRawTensor().data().asInts().read(buffer32S.array());
		    else {
		    	int[] values = new int[size];
			    tensor.asRawTensor().data().asInts().read(values);
			    buffer32S.put(values);
		    }
		    return mat;
		case DT_INT64:
			break;
		case DT_INT8:
			break;
		case DT_INVALID:
			break;
		case DT_QINT16:
			break;
		case DT_QINT32:
			break;
		case DT_QINT8:
			break;
		case DT_QUINT16:
			break;
		case DT_QUINT8:
			break;
		case DT_RESOURCE:
			break;
		case DT_STRING:
			break;
		case DT_UINT16:
			break;
		case DT_UINT32:
			break;
		case DT_UINT64:
			break;
		case DT_UINT8:
			mat = new Mat(dims[0], dims[1], opencv_core.CV_8UC(dims[2]));
		    ByteBuffer buffer8U = mat.createBuffer();
		    if (buffer8U.hasArray())
			    tensor.asRawTensor().data().read(buffer8U.array());
		    else {
		    	byte[] values = new byte[size];
			    tensor.asRawTensor().data().read(values);
			    buffer8U.put(values);
		    }
		    return mat;
		case DT_VARIANT:
			break;
		case UNRECOGNIZED:
			break;
		default:
			break;
	    }
	    throw new UnsupportedOperationException("Unsupported Tensor to Mat conversion for DataType " + tensor.dataType());
	}
	
	/**
	 * Convert a tensor to a list of Mats, one for each index along the batch (first) axis.
	 * Note: Currently only 32-bit float tensors are supported.
	 * @param tensor
	 * @return
	 */
	public static List<Mat> convertToMats(Tensor tensor) {
		if (tensor instanceof FloatNdArray)
			return convertNdArrayToMats((FloatNdArray)tensor);
		else
			throw new UnsupportedOperationException("Tensor must be instance of FloatNdArray for conversion to Mat! Here tensor is " + tensor);
	}
	
	/**
	 * Convert an NdArray to a list of Mats, one for each index along the batch (first) axis.
	 * @param tensor
	 * @return
	 */
	public static List<Mat> convertNdArrayToMats(FloatNdArray tensor) {
		
		var list = new ArrayList<Mat>();
		
		var shape = tensor.shape();
		
		long[] sizes = shape.asArray();
		Index[] inds = Stream.of(sizes).map(s -> Indices.all()).toArray(Index[]::new);
		int nDim = sizes.length;
		long n = sizes[0];
		int h = nDim > 1 ? (int)sizes[1] : 1;
		int w = nDim > 2 ? (int)sizes[2] : 1;
		int channels = nDim > 3 ? (int)sizes[3] : 1;
	    for (long i = 0; i < n; i++) {
	    	
	    	inds[0] = Indices.at(i);
	    	var slice = tensor.slice(inds);
	    	
	    	var mat = new Mat(h, w, opencv_core.CV_32FC(channels));
	    	FloatBuffer buffer = mat.createBuffer();
	    	FloatDataBuffer dataBuffer = DataBuffers.of(buffer);
	    	
	    	slice.read(dataBuffer);
	    	
	    	list.add(mat);
	    }
		
	    return list;
	}

	/**
	 * Create an {@link ImageOp} to run a TensorFlow model with a single image input and output, 
	 * optionally specifying the input tile width and height.
	 * 
	 * @param modelPath
	 * @param tileWidth input tile width; ignored if &le; 0
	 * @param tileHeight input tile height; ignored if &le; 0
	 * @param padding amount of padding to add to each request
	 * @return the {@link ImageOp}
	 * @throws IllegalArgumentException if the model path is not a directory
	 */
	public static ImageOp createOp(String modelPath, int tileWidth, int tileHeight, Padding padding) throws IllegalArgumentException {
		return createOp(modelPath, tileWidth, tileHeight, padding, null);
	}
	
	/**
	 * Create an {@link ImageOp} to run a TensorFlow model with a single image input and output, 
	 * optionally specifying the input tile width and height.
	 * 
	 * @param modelPath
	 * @param tileWidth input tile width; ignored if &le; 0
	 * @param tileHeight input tile height; ignored if &le; 0
	 * @param padding amount of padding to add to each request
	 * @param outputName optional name of the node to use for output (may be null)
	 * @return the {@link ImageOp}
	 * @throws IllegalArgumentException if the model path is not a directory
	 * @deprecated use {@link #createDnnModel(String)} instead, then ImageOps to build an op
	 */
	@Deprecated
	public static ImageOp createOp(String modelPath, int tileWidth, int tileHeight, Padding padding, String outputName) throws IllegalArgumentException {
		var file = new File(modelPath);
		if (!file.isDirectory()) {
			logger.error("Invalid model path, not a directory! {}", modelPath);
			throw new IllegalArgumentException("Model path should be a directory!");
		}
		return new TensorFlowOp(modelPath, tileWidth, tileHeight, padding, outputName);
	}
	
	
	
	
	/**
	 * Create a {@link DnnModel} for TensorFlow by reading a specified model file.
	 * @param modelPath
	 * @return
	 * @throws IOException
	 */
	public static DnnModel<Tensor> createDnnModel(String modelPath) throws IOException {
		try {
			return createDnnModel(GeneralTools.toURI(modelPath));
		} catch (URISyntaxException e) {
			throw new IOException(e);
		}
	}
	
	/**
	 * Create a {@link DnnModel} for TensorFlow by reading a specified URI.
	 * @param uri
	 * @return
	 */
	public static DnnModel<Tensor> createDnnModel(URI uri) {
		return new TensorFlowDnnModel(uri);
	}
	

}
