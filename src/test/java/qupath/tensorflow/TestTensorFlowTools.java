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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.junit.jupiter.api.Test;

import qupath.opencv.tools.OpenCVTools;

@SuppressWarnings("javadoc")
public class TestTensorFlowTools {
	
	@Test
	public void test_convertFloat32() {
		
		int rows = 4;
		int cols = 5;
		int channels = 2;
		float[] values = new float[] {1f, 2.5f, 4f};
		int n = values.length;
		
		try (var scope = new PointerScope()) {
			
			var mats = new Mat[n];
			for (int i = 0; i < n; i++) {
				mats[i] = new Mat(rows, cols, opencv_core.CV_32FC(channels), Scalar.all(values[i]));
			}
			
			var tensor = TensorFlowTools.convertToTensorFloat32(mats);
			
			var shape = tensor.shape();
			
			assertArrayEquals(shape.asArray(), new long[] {n, rows, cols, channels});
			
			// Check values are correct
			long[] inds = new long[4];
			for (long b = 0; b < n; b++) {
				inds[0] = b;
				for (long r = 0; r < rows; r++) {
					inds[1] = r;
					for (long c = 0; c < cols; c++) {
						inds[2] = c;
						for (long channel = 0; channel < channels; channel++) {
							inds[3] = channel;
							float v = tensor.getFloat(inds);
							assertEquals(v, values[(int)b], 1e-6);
						}
					}
				}
			}
			
			// Convert back to Mats
			var mats2 = TensorFlowTools.convertToMats(tensor);
			assertEquals(mats2.size(), mats.length);
			for (int i = 0; i < mats2.size(); i++) {
				var m2 = mats2.get(i);
				assertEquals(m2.rows(), rows);
				assertEquals(m2.cols(), cols);
				assertEquals(m2.channels(), channels);
				
				var m = mats[i];
				assertArrayEquals(OpenCVTools.extractDoubles(m), OpenCVTools.extractDoubles(m2), 1e-6);
			}
		}
		
	}

}
