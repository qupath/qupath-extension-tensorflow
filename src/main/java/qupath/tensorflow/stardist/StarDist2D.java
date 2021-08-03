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

package qupath.tensorflow.stardist;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.locationtech.jts.algorithm.Centroid;
import org.locationtech.jts.algorithm.locate.SimplePointInAreaLocator;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.Location;
import org.locationtech.jts.geom.Polygon;
import org.locationtech.jts.index.strtree.STRtree;
import org.locationtech.jts.simplify.VWSimplifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.analysis.features.ObjectMeasurements;
import qupath.lib.analysis.features.ObjectMeasurements.Compartments;
import qupath.lib.analysis.features.ObjectMeasurements.Measurements;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.images.servers.ColorTransforms.ColorTransform;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.images.servers.PixelType;
import qupath.lib.images.servers.TransformedServerBuilder;
import qupath.lib.objects.CellTools;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.Padding;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.dnn.OpenCVDNN;
import qupath.opencv.ops.ImageDataOp;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;

/**
 * Cell detection based on the following method:
 * <pre>
 *   Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.
 *     "Cell Detection with Star-convex Polygons."
 *   <i>International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)</i>, Granada, Spain, September 2018.
 * </pre>
 * See the main repo at https://github.com/mpicbg-csbd/stardist
 * <p>
 * Very much inspired by stardist-imagej at https://github.com/mpicbg-csbd/stardist-imagej but re-written from scratch to use OpenCV and 
 * adapt the method of converting predictions to contours (very slightly) to be more QuPath-friendly.
 * <p>
 * Models are expected in the same format as required by the Fiji plugin.
 * 
 * @author Pete Bankhead (this implementation, but based on the others)
 */
public class StarDist2D {

	private final static Logger logger = LoggerFactory.getLogger(StarDist2D.class);
	
	/**
	 * Builder to help create a {@link StarDist2D} with custom parameters.
	 */
	public static class Builder {
		
		private boolean doLog;
		
		private int nThreads = -1;
		
		private String modelPath = null;
		private ImageOp predictionOp = null;
		private ColorTransform[] channels = new ColorTransform[0];
		
		private double threshold = 0.5;
		
		private double simplifyDistance = 1.4;
		private double cellExpansion = Double.NaN;
		private double cellConstrainScale = Double.NaN;
		private boolean ignoreCellOverlaps = false;

		private double pixelSize = Double.NaN;
				
		private int tileWidth = 1024;
		private int tileHeight = 1024;
		
		private Function<ROI, PathObject> creatorFun;
		
		private PathClass globalPathClass;
		private Map<Integer, PathClass> classifications;
		
		private boolean measureShape = false;
		private Collection<Compartments> compartments = Arrays.asList(Compartments.values());
		private Collection<Measurements> measurements;
		
		private boolean keepClassifiedBackground = false;
		
		private boolean constrainToParent = true;
		
		private int pad = 32;
		
		private List<ImageOp> ops = new ArrayList<>();
		
		private boolean includeProbability = false;
		
		private Builder(String modelPath) {
			this.modelPath = modelPath;
			this.ops.add(ImageOps.Core.ensureType(PixelType.FLOAT32));
		}
		
		private Builder(ImageOp predictionOp) {
			this.predictionOp = predictionOp;
			this.ops.add(ImageOps.Core.ensureType(PixelType.FLOAT32));
		}
		
		/**
		 * Probability threshold to apply for detection, between 0 and 1.
		 * @param threshold
		 * @return this builder
		 * @see #includeProbability(boolean)
		 */
		public Builder threshold(double threshold) {
			this.threshold = threshold;
			return this;
		}
		
		/**
		 * Add preprocessing operations, if required.
		 * @param ops
		 * @return this builder
		 */
		public Builder preprocess(ImageOp... ops) {
			for (var op : ops)
				this.ops.add(op);
			return this;
		}
		
		/**
		 * Request that progress is logged. If this is not specified, progress is only logged at the DEBUG level.
		 * @return this builder
		 */
		public Builder doLog() {
			this.doLog = true;
			return this;
		}
		
		/**
		 * Customize the extent to which contours are simplified.
		 * Simplification reduces the number of vertices, which in turn can reduce memory requirements and 
		 * improve performance.
		 * <p>
		 * Implementation note: this currently uses the Visvalingam-Whyatt algorithm.
		 * 
		 * @param distance simplify distance threshold; set &le; 0 to turn off additional simplification
		 * @return this builder
		 */
		public Builder simplify(double distance) {
			this.simplifyDistance = distance;
			return this;
		}
		
		/**
		 * Specify channels. Useful for detecting nuclei for one channel 
		 * within a multi-channel image, or potentially for trained models that 
		 * support multi-channel input.
		 * @param channels 0-based indices of the channels to use
		 * @return this builder
		 */
		public Builder channels(int... channels) {
			return channels(Arrays.stream(channels)
					.mapToObj(c -> ColorTransforms.createChannelExtractor(c))
					.toArray(ColorTransform[]::new));
		}
		
		/**
		 * Specify channels by name. Useful for detecting nuclei for one channel 
		 * within a multi-channel image, or potentially for trained models that 
		 * support multi-channel input.
		 * @param channels 0-based indices of the channels to use
		 * @return this builder
		 */
		public Builder channels(String... channels) {
			return channels(Arrays.stream(channels)
					.map(c -> ColorTransforms.createChannelExtractor(c))
					.toArray(ColorTransform[]::new));
		}
		
		/**
		 * Define the channels (or color transformers) to apply to the input image.
		 * <p>
		 * This makes it possible to supply color deconvolved channels, for example.
		 * @param channels
		 * @return this builder
		 */
		public Builder channels(ColorTransform... channels) {
			this.channels = channels.clone();
			return this;
		}
		
		/**
		 * Amount by which to expand detected nuclei to approximate the cell area.
		 * Units are the same as for the {@link PixelCalibration} of the input image.
		 * <p>
		 * Warning! This is rather experimental, relying heavily on JTS and a convoluted method of 
		 * resolving overlaps using a Voronoi tessellation.
		 * <p>
		 * In short, be wary.
		 * @param distance
		 * @return this builder
		 */
		public Builder cellExpansion(double distance) {
			this.cellExpansion = distance;
			return this;
		}
		
		/**
		 * Constrain any cell expansion defined using {@link #cellExpansion(double)} based upon 
		 * the nucleus size. Only meaningful for values &gt; 1; the nucleus is expanded according 
		 * to the scale factor, and used to define the maximum permitted cell expansion.
		 * 
		 * @param scale
		 * @return this builder
		 */
		public Builder cellConstrainScale(double scale) {
			this.cellConstrainScale = scale;
			return this;
		}
		
		/**
		 * Create annotations rather than detections (the default).
		 * If cell expansion is not zero, the nucleus will be included as a child object.
		 * 
		 * @return this builder
		 */
		public Builder createAnnotations() {
			this.creatorFun = r -> PathObjects.createAnnotationObject(r);
			return this;
		}
		
		/**
		 * Specify a mapping between StarDist predicted classifications (if available) and QuPath classifications.
		 * 
		 * @param classifications
		 * @return this builder
		 * 
		 * @see #classify(PathClass)
		 */
		public Builder classifications(Map<Integer, PathClass> classifications) {
			this.classifications = new HashMap<>(classifications);
			return this;
		}

		/**
		 * Specify a mapping between StarDist predicted classifications (if available) and QuPath classification names.
		 * This is a convenience method that creates {@link PathClass} objects from Strings, then passes them to {@link #classifications(Map)}.
		 * 
		 * @param classifications
		 * @return this builder
		 */
		public Builder classificationNames(Map<Integer, String> classifications) {
			return classifications(classifications.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> PathClassFactory.getPathClass(e.getValue()))));
		}
		
		/**
		 * When using {@link #classifications(Map)}, optionally keep objects classified as background (i.e. 0).
		 * The default is to remove such objects immediately, before resolving overlaps.
		 * @param keep
		 * @return this builder
		 */
		public Builder keepClassifiedBackground(boolean keep) {
			this.keepClassifiedBackground = keep;
			return this;
		}

		/**
		 * Request that a classification is applied to all created objects.
		 * Note that if a StarDist model supporting classifications is used and {@link #classifications(Map)} is specified, 
		 * the StarDist prediction will take precedence. The classification specified here will be applied only to objects 
		 * that have not already been classified based upon the prediction and mapping.
		 * 
		 * @param pathClass
		 * @return this builder
		 * 
		 * @see #classifications(Map)
		 * @see #classificationNames(Map)
		 */
		public Builder classify(PathClass pathClass) {
			this.globalPathClass = pathClass;
			return this;
		}
		
		/**
		 * Request that a classification is applied to all created objects.
		 * This is a convenience method that get a {@link PathClass} from  {@link PathClassFactory}.
		 * 
		 * @param pathClassName
		 * @return this builder
		 * @see #classifications(Map)
		 * @see #classificationNames(Map)
		 */
		public Builder classify(String pathClassName) {
			return classify(PathClassFactory.getPathClass(pathClassName, (Integer)null));
		}
		
		/**
		 * If true, ignore overlaps when computing cell expansion.
		 * @param ignore
		 * @return this builder
		 */
		public Builder ignoreCellOverlaps(boolean ignore) {
			this.ignoreCellOverlaps = ignore;
			return this;
		}
		
		/**
		 * If true, constrain nuclei and cells to any parent annotation (default is true).
		 * @param constrainToParent
		 * @return this builder
		 */
		public Builder constrainToParent(boolean constrainToParent) {
			this.constrainToParent = constrainToParent;
			return this;
		}
		
		/**
		 * Specify the number of threads to use for processing.
		 * If you encounter problems, setting this to 1 may help to resolve them by preventing 
		 * multithreading.
		 * @param nThreads
		 * @return this builder
		 */
		public Builder nThreads(int nThreads) {
			this.nThreads = nThreads;
			return this;
		}
		
		/**
		 * Request default intensity measurements are made for all available cell compartments.
		 * @return this builder
		 */
		public Builder measureIntensity() {
			this.measurements = Arrays.asList(
					Measurements.MEAN,
					Measurements.MEDIAN,
					Measurements.MIN,
					Measurements.MAX,
					Measurements.STD_DEV);
			return this;
		}
		
		/**
		 * Request specified intensity measurements are made for all available cell compartments.
		 * @param measurements the measurements to make
		 * @return this builder
		 */
		public Builder measureIntensity(Collection<Measurements> measurements) {
			this.measurements = new ArrayList<>(measurements);
			return this;
		}
		
		/**
		 * Request shape measurements are made for the detected cell or nucleus.
		 * @return this builder
		 */
		public Builder measureShape() {
			measureShape = true;
			return this;
		}
		
		/**
		 * Specify the compartments within which intensity measurements are made.
		 * Only effective if {@link #measureIntensity()} and {@link #cellExpansion(double)} have been selected.
		 * @param compartments cell compartments for intensity measurements
		 * @return this builder
		 */
		public Builder compartments(Compartments...compartments) {
			this.compartments = Arrays.asList(compartments);
			return this;
		}
		
		/**
		 * Optionally include the prediction probability as a measurement for the object.
		 * This can be helpful if detection is applied with a low (generous) probability threshold, 
		 * with the intention of filtering out less likely detections later.
		 * 
		 * @param include true if the probability should be included, false otherwise
		 * @return this builder
		 * @see #threshold(double)
		 */
		public Builder includeProbability(boolean include) {
			this.includeProbability = include;
			return this;
		}
		
		/**
		 * Resolution at which the cell detection should be run.
		 * The units depend upon the {@link PixelCalibration} of the input image.
		 * <p>
		 * The default is to use the full resolution of the input image.
		 * <p>
		 * For an image calibrated in microns, the recommended default is approximately 0.5.
		 * 
		 * @param pixelSize
		 * @return this builder
		 */
		public Builder pixelSize(double pixelSize) {
			this.pixelSize = pixelSize;
			return this;
		}
		
		/**
		 * Size in pixels of a tile used for detection.
		 * Note that tiles are independently normalized, and therefore tiling can impact 
		 * the results. Default is 1024.
		 * @param tileSize
		 * @return this builder
		 */
		public Builder tileSize(int tileSize) {
			return tileSize(tileSize, tileSize);
		}
		
		/**
		 * Size in pixels of a tile used for detection.
		 * Note that tiles are independently normalized, and therefore tiling can impact 
		 * the results. Default is 1024.
		 * @param tileWidth
		 * @param tileHeight
		 * @return this builder
		 */
		public Builder tileSize(int tileWidth, int tileHeight) {
			this.tileWidth = tileWidth;
			this.tileHeight = tileHeight;
			return this;
		}
		
		/**
		 * Amount to pad tiles to reduce boundary artifacts.
		 * @param pad padding in pixels; width and height of tiles will be increased by pad x 2.
		 * @return this builder
		 */
		public Builder padding(int pad) {
			this.pad = pad;
			return this;
		}
				
		/**
		 * Apply percentile normalization to the input image channels.
		 * <p>
		 * Note that this can be used in combination with {@link #preprocess(ImageOp...)}, 
		 * in which case the order in which the operations are applied depends upon the order 
		 * in which the methods of the builder are called.
		 * <p>
		 * Warning! This is applied on a per-tile basis. This can result in artifacts and false detections 
		 * without background/constant regions. 
		 * Consider using {@link #inputAdd(double...)} and {@link #inputScale(double...)} as alternative 
		 * normalization strategies, if appropriate constants can be determined to apply globally.
		 * 
		 * @param min minimum percentile
		 * @param max maximum percentile
		 * @return this builder
		 */
		public Builder normalizePercentiles(double min, double max) {
			this.ops.add(ImageOps.Normalize.percentile(min, max));
			return this;
		}
		
		/**
		 * Add an offset as a preprocessing step.
		 * Usually the value will be negative. Along with {@link #inputScale(double...)} this can be used as an alternative (global) normalization.
		 * <p>
		 * Note that this can be used in combination with {@link #preprocess(ImageOp...)}, 
		 * in which case the order in which the operations are applied depends upon the order 
		 * in which the methods of the builder are called.
		 * 
		 * @param values either a single value to add to all channels, or an array of values equal to the number of channels
		 * @return this builder
		 */
		public Builder inputAdd(double... values) {
			this.ops.add(ImageOps.Core.add(values));
			return this;
		}
		
		/**
		 * Multiply by a scale factor as a preprocessing step.
		 * Along with {@link #inputAdd(double...)} this can be used as an alternative (global) normalization.
		 * <p>
		 * Note that this can be used in combination with {@link #preprocess(ImageOp...)}, 
		 * in which case the order in which the operations are applied depends upon the order 
		 * in which the methods of the builder are called.
		 * 
		 * @param values either a single value to add to all channels, or an array of values equal to the number of channels
		 * @return this builder
		 */
		public Builder inputScale(double... values) {
			this.ops.add(ImageOps.Core.subtract(values));
			return this;
		}
		
		/**
		 * Create a {@link StarDist2D}, all ready for detection.
		 * @return
		 */
		public StarDist2D build() {
			var stardist = new StarDist2D();
			
			var padding = pad > 0 ? Padding.symmetric(pad) : Padding.empty();
			var mergedOps = new ArrayList<>(ops);
			var mlOp = this.predictionOp;
			if (mlOp == null) {
				var file = new File(modelPath);
				if (!file.exists()) {
					throw new IllegalArgumentException("I couldn't find the model file " + file.getAbsolutePath());
					// TODO: In the future, search within the user directory
				}
				if (file.isFile()) {
					try {
						var dnn = new OpenCVDNN.Builder(file.toURI())
								.build();
						mlOp = ImageOps.ML.dnn(dnn, tileWidth, tileHeight, padding);
						logger.debug("Loaded model {} with OpenCV DNN", modelPath);
					} catch (Exception e) {
						logger.error("Unable to load model file with OpenCV. If you intended to use TensorFlow, you need to have it on the classpath & provide the "
								+ "path to the directory, not the .pb file.");
						logger.error(e.getLocalizedMessage(), e);
						throw new RuntimeException("Unable to load StarDist model from " + modelPath, e);
					}
				} else {
					try {
						var clsTF = Class.forName("qupath.tensorflow.TensorFlowTools");
						var method = clsTF.getMethod("createOp", String.class, int.class, int.class, Padding.class);
						mlOp = (ImageOp)method.invoke(null, modelPath, tileWidth, tileHeight, padding);
						logger.debug("Loaded model {} with TensorFlow", modelPath);
//						mlOp = TensorFlowTools.createOp(modelPath, tileWidth, tileHeight, padding);				
					} catch (Exception e) {
						logger.error("Unable to load TensorFlow with reflection - are you sure it is available and on the classpath?");
						logger.error(e.getLocalizedMessage(), e);
						throw new RuntimeException("Unable to load StarDist model from " + modelPath, e);
					}
				}
			}
			mergedOps.add(mlOp);
			mergedOps.add(ImageOps.Core.ensureType(PixelType.FLOAT32));
			
			stardist.op = ImageOps.buildImageDataOp(channels)
					.appendOps(mergedOps.toArray(ImageOp[]::new));
			stardist.threshold = threshold;
			stardist.pixelSize = pixelSize;
			stardist.cellConstrainScale = cellConstrainScale;
			stardist.cellExpansion = cellExpansion;
			stardist.tileWidth = tileWidth-pad*2;
			stardist.tileHeight = tileHeight-pad*2;
			stardist.includeProbability = includeProbability;
			stardist.ignoreCellOverlaps = ignoreCellOverlaps;
			stardist.measureShape = measureShape;
			stardist.doLog = doLog;
			stardist.simplifyDistance = simplifyDistance;
			stardist.nThreads = nThreads;
			stardist.constrainToParent = constrainToParent;
			stardist.creatorFun = creatorFun;
			stardist.globalPathClass = globalPathClass;
			stardist.classifications = classifications;
			stardist.keepClassifiedBackground = keepClassifiedBackground;
			
			stardist.compartments = new LinkedHashSet<>(compartments);
			
			if (measurements != null)
				stardist.measurements = new LinkedHashSet<>(measurements);
			else
				stardist.measurements = Collections.emptyList();
			
			return stardist;
		}
		
	}
	
	private boolean doLog = false;
	
	private double simplifyDistance = 1.4;
	
	private double threshold;
	
	private ImageDataOp op;
	private double pixelSize;
	private double cellExpansion;
	private double cellConstrainScale;
	private boolean ignoreCellOverlaps;
	
	private Function<ROI, PathObject> creatorFun;
	private PathClass globalPathClass;
	private Map<Integer, PathClass> classifications;
	private boolean keepClassifiedBackground = false;
	
	private boolean constrainToParent = true;
	
	private int nThreads = -1;
	
	private boolean includeProbability = false;
	
	private int tileWidth = 1024;
	private int tileHeight = 1024;

	private boolean measureShape = false;

	private Collection<ObjectMeasurements.Compartments> compartments;
	private Collection<ObjectMeasurements.Measurements> measurements;
	
	
	
	
	/**
	 * Detect cells within one or more parent objects, firing update events upon completion.
	 * 
	 * @param imageData the image data containing the object
	 * @param parents the parent objects; existing child objects will be removed, and replaced by the detected cells
	 */
	public void detectObjects(ImageData<BufferedImage> imageData, Collection<? extends PathObject> parents) {
		runInPool(() -> detectObjectsImpl(imageData, parents));		
	}

	/**
	 * Detect cells within a parent object.
	 * 
	 * @param imageData the image data containing the object
	 * @param parent the parent object; existing child objects will be removed, and replaced by the detected cells
	 * @param fireUpdate if true, a hierarchy update will be fired on completion
	 */
	public void detectObjects(ImageData<BufferedImage> imageData, PathObject parent, boolean fireUpdate) {
		runInPool(() -> detectObjectsImpl(imageData, parent, fireUpdate));
	}
	
	/**
	 * Optionally submit runnable to a thread pool. This limits the parallelization used by parallel streams.
	 * @param runnable
	 */
	private void runInPool(Runnable runnable) {
		if (nThreads > 0) {
			if (nThreads == 1)
				log("Processing with {} thread", nThreads);
			else
				log("Processing with {} threads", nThreads);
			// Using an outer thread poll impacts any parallel streams created inside
			var pool = new ForkJoinPool(nThreads);
			try {
				pool.submit(() -> runnable.run());
			} finally {
				pool.shutdown();
				try {
					pool.awaitTermination(24, TimeUnit.HOURS);
				} catch (InterruptedException e) {
					logger.warn("Process was interrupted! " + e.getLocalizedMessage(), e);
				}
			}
		} else {
			runnable.run();	
		}
	}
	
		
	private void detectObjectsImpl(ImageData<BufferedImage> imageData, Collection<? extends PathObject> parents) {

		if (parents.isEmpty())
			return;
		if (parents.size() == 1) {
			detectObjects(imageData, parents.iterator().next(), true);
			return;
		}
		log("Processing {} parent objects", parents.size());
		if (nThreads >= 0)
			parents.stream().forEach(p -> detectObjects(imageData, p, false));
		else
			parents.parallelStream().forEach(p -> detectObjects(imageData, p, false));
		// Fire a globel update event
		imageData.getHierarchy().fireHierarchyChangedEvent(imageData.getHierarchy());
	}
	
	
	/**
	 * Detect cells within a parent object.
	 * 
	 * @param imageData the image data containing the object
	 * @param parent the parent object; existing child objects will be removed, and replaced by the detected cells
	 * @param fireUpdate if true, a hierarchy update will be fired on completion
	 */
	private void detectObjectsImpl(ImageData<BufferedImage> imageData, PathObject parent, boolean fireUpdate) {
		Objects.nonNull(parent);
		// Lock early, so the user doesn't make modifications
		parent.setLocked(true);
		
		List<PathObject> detections = detectObjects(imageData, parent.getROI());		
		
		parent.clearPathObjects();
		parent.addPathObjects(detections);
		if (fireUpdate)
			imageData.getHierarchy().fireHierarchyChangedEvent(imageData.getHierarchy(), parent);
	}
	
	
	
	/**
	 * Detect cells within a {@link ROI}.
	 * @param imageData image to which the ROI belongs
	 * @param roi region of interest which which to detect cells. If null, the entire image will be used.
	 * @return the detected objects. Note that these will not automatically be added to the object hierarchy.
	 */
	public List<PathObject> detectObjects(ImageData<BufferedImage> imageData, ROI roi) {

		var resolution = imageData.getServer().getPixelCalibration();
		if (Double.isFinite(pixelSize) && pixelSize > 0) {
			double downsample = pixelSize / resolution.getAveragedPixelSize().doubleValue();
			resolution = resolution.createScaledInstance(downsample, downsample);
		}
		var opServer = ImageOps.buildServer(imageData, op, resolution, tileWidth, tileHeight);
		
		RegionRequest request;
		if (roi == null)
			request = RegionRequest.createInstance(opServer);
		else
			request = RegionRequest.createInstance(
				opServer.getPath(),
				opServer.getDownsampleForResolution(0),
				roi);

		var tiles = opServer.getTileRequestManager().getTileRequests(request);
		var mask = roi == null ? null : roi.getGeometry();
				
		// Detect all potential nuclei
		var server = imageData.getServer();
		var cal = server.getPixelCalibration();
		double expansion = cellExpansion / cal.getAveragedPixelSize().doubleValue();
		var plane = roi.getImagePlane();
//		var detections = tiles.parallelStream()
//			.flatMap(t -> detectObjectsForTile(op, imageData, t.getRegionRequest(), tiles.size() > 1, mask).stream())
//			.map(n -> convertToObject(n, plane, expansion, mask))
//			.collect(Collectors.toList());

		if (tiles.size() > 1)
			log("Detecting nuclei for {} tiles", tiles.size());
		else
			log("Detecting nuclei");
		var nuclei = tiles.parallelStream()
				.flatMap(t -> detectObjectsForTile(op, imageData, t.getRegionRequest(), tiles.size() > 1, mask).stream())
				.collect(Collectors.toList());
		
		// Filter nuclei again if we need to for resolving tile overlaps
		if (tiles.size() > 1) {
			log("Resolving nucleus overlaps");
			nuclei = filterNuclei(nuclei);
		}
		
		// Convert to detections, dilating to approximate cells if necessary
		// Drop cells if they fail (rather than catastropically give up)
		var detections = nuclei.parallelStream()
				.map(n -> {
					try {
						return convertToObject(n, plane, expansion, constrainToParent ? mask : null);
					} catch (Exception e) {
						logger.warn("Error converting to object: " + e.getLocalizedMessage(), e);
						return null;
					}
				}).filter(n -> n != null)
				.collect(Collectors.toList());
		
		// Resolve cell overlaps, if needed
		if (expansion > 0 && !ignoreCellOverlaps) {
			log("Resolving cell overlaps");
			if (creatorFun != null) {
				// It's awkward, but we need to temporarily convert to cells and back
				var cells = detections.stream().map(c -> objectToCell(c)).collect(Collectors.toList());
				cells = CellTools.constrainCellOverlaps(cells);
				detections = cells.stream().map(c -> cellToObject(c, creatorFun)).collect(Collectors.toList());
			} else
				detections = CellTools.constrainCellOverlaps(detections);
		}
		
		// Add shape measurements, if needed
		if (measureShape)
			detections.parallelStream().forEach(c -> ObjectMeasurements.addShapeMeasurements(c, cal));
		
		// Add intensity measurements, if needed
		if (!detections.isEmpty() && !measurements.isEmpty()) {
			log("Making measurements");
			var stains = imageData.getColorDeconvolutionStains();
			var builder = new TransformedServerBuilder(server);
			if (stains != null) {
				List<Integer> stainNumbers = new ArrayList<>();
				for (int s = 1; s <= 3; s++) {
					if (!stains.getStain(s).isResidual())
						stainNumbers.add(s);
				}
				builder.deconvolveStains(stains, stainNumbers.stream().mapToInt(i -> i).toArray());
			}
			
			var server2 = builder.build();
			double downsample = resolution.getAveragedPixelSize().doubleValue() / cal.getAveragedPixelSize().doubleValue();
			
			detections.parallelStream().forEach(cell -> {
				try {
					ObjectMeasurements.addIntensityMeasurements(server2, cell, downsample, measurements, compartments);					
				} catch (IOException e) {
					log(e.getLocalizedMessage(), e);
				}
			});
			
		}
		
		log("Detected {} cells", detections.size());

		return detections;
	}
	
	
	private static PathObject objectToCell(PathObject pathObject) {
		ROI roiNucleus = null;
		var children = pathObject.getChildObjects();
		if (children.size() == 1)
			roiNucleus = children.iterator().next().getROI();
		else if (children.size() > 1)
			throw new IllegalArgumentException("Cannot convert object with multiple child objects to a cell!");
		return PathObjects.createCellObject(pathObject.getROI(), roiNucleus, pathObject.getPathClass(), pathObject.getMeasurementList());
	}
	
	private static PathObject cellToObject(PathObject cell, Function<ROI, PathObject> creator) {
		var parent = creator.apply(cell.getROI());
		var nucleusROI = cell instanceof PathCellObject ? ((PathCellObject)cell).getNucleusROI() : null;
		if (nucleusROI != null) {
			var nucleus = creator.apply(nucleusROI);
			nucleus.setPathClass(cell.getPathClass());
			parent.addPathObject(nucleus);
		}
		parent.setPathClass(cell.getPathClass());
		var cellMeasurements = cell.getMeasurementList();
		if (!cellMeasurements.isEmpty()) {
			try (var ml = parent.getMeasurementList()) {
				for (int i = 0; i < cellMeasurements.size(); i++)
					ml.addMeasurement(cellMeasurements.getMeasurementName(i), cellMeasurements.getMeasurementValue(i));
			}
		}
		return parent;
	}
	
	
	
	private void log(String message, Object... arguments) {
		if (doLog)
			logger.info(message, arguments);
		else
			logger.debug(message, arguments);			
	}
	
	
	private PathObject convertToObject(PotentialNucleus nucleus, ImagePlane plane, double cellExpansion, Geometry mask) {
		var geomNucleus = simplify(nucleus.geometry);
		PathObject pathObject;
		if (cellExpansion > 0) {
			var geomCell = CellTools.estimateCellBoundary(geomNucleus, cellExpansion, cellConstrainScale);
			if (mask != null)
				geomCell = GeometryTools.attemptOperation(geomCell, g -> g.intersection(mask));
			geomCell = simplify(geomCell);
			var roiCell = GeometryTools.geometryToROI(geomCell, plane);
			var roiNucleus = GeometryTools.geometryToROI(geomNucleus, plane);
			if (creatorFun == null)
				pathObject = PathObjects.createCellObject(roiCell, roiNucleus, null, null);
			else {
				pathObject = creatorFun.apply(roiCell);
				if (roiNucleus != null) {
					pathObject.addPathObject(creatorFun.apply(roiNucleus));
				}
			}
		} else {
			if (mask != null) {
				geomNucleus = GeometryTools.attemptOperation(geomNucleus, g -> g.intersection(mask));
			}
			var roiNucleus = GeometryTools.geometryToROI(geomNucleus, plane);
			if (creatorFun == null)
				pathObject = PathObjects.createDetectionObject(roiNucleus);
			else
				pathObject = creatorFun.apply(roiNucleus);
		}
		if (includeProbability) {
        	try (var ml = pathObject.getMeasurementList()) {
        		ml.putMeasurement("Detection probability", nucleus.getProbability());
        	}
        }
		
		// Set classification, if available
		PathClass pathClass;
		if (classifications == null)
			pathClass = globalPathClass;
		else
			pathClass = classifications.getOrDefault(nucleus.getClassification(), globalPathClass);
		
		if (pathClass != null && pathClass.isValid())
			pathObject.setPathClass(pathClass);
		return pathObject;
	}
	
	
	private Geometry simplify(Geometry geom) {
		if (simplifyDistance <= 0)
			return geom;
		try {
			return VWSimplifier.simplify(geom, simplifyDistance);
		} catch (Exception e) {
			return geom;
		}
	}
	
	
	private static int[] range(int startInclusive, int endExclusive) {
		int n = endExclusive - startInclusive;
		int[] output = new int[n];
		for (int i = 0; i < n; i++)
			output[i] = startInclusive + i;
		return output;
	}
	
	
	private static Mat extractChannels(Mat mat, int... channels) {
		Mat output;
		int n = channels.length;
		if (n == 0) {
			output = new Mat();			
		} else if (n == 1) {
			output = new Mat();
			opencv_core.extractChannel(mat, output, channels[0]);
		} else {
			int[] pairs = new int[n * 2];
			for (int i = 0; i < n; i++) {
				pairs[i*2] = channels[i];
				pairs[i*2+1] = i;
			}
			output = new Mat(mat.rows(), mat.cols(), opencv_core.CV_MAKE_TYPE(mat.depth(), n));
			opencv_core.mixChannels(mat, 1, output, 1, pairs, n);
		}
		return output;
	}
	
	
	
	private List<PotentialNucleus> detectObjectsForTile(ImageDataOp op, ImageData<BufferedImage> imageData, RegionRequest request, boolean excludeOnBounds, Geometry mask) {

		List<PotentialNucleus> nuclei;
		
		try (@SuppressWarnings("unchecked")
		var scope = new PointerScope()) {
			Mat mat;
			try {
				mat = op.apply(imageData, request);
			} catch (IOException e) {
				logger.error(e.getLocalizedMessage(), e);
				return Collections.emptyList();
			}
			
//			OpenCVTools.matToImagePlus(mat, "Prediction " + request).show();
			
			// Split channels to extract probability, ray and (possibly) classification images
			int nChannels = mat.channels();
			int nClassifications = classifications == null ? 0 : classifications.size();
			int nRays = nChannels - 1 - nClassifications;
			Mat matProb = extractChannels(mat, 0);
			Mat matRays = extractChannels(mat, range(1, nRays+1));
			Mat matClassifications = nClassifications == 0 ? null : extractChannels(mat, range(nRays+1, nChannels));
			
			// Depending upon model export, we might have a half resolution prediction that needs to be rescaled
			long inputWidth = Math.round(request.getWidth() / request.getDownsample());
			long inputHeight = Math.round(request.getHeight() / request.getDownsample());
			double scaleX = Math.round(inputWidth / mat.cols());
			double scaleY = Math.round(inputHeight / mat.rows());
			if (scaleX != 1.0 || scaleY != 1.0) {
				if (scaleX != 2.0 || scaleY != 2.0)
					logger.warn("Unexpected StarDist rescaling x={}, y={}", scaleX, scaleY);
				else
					logger.debug("StarDist rescaling x={}, y={}", scaleX, scaleY);
			}
			
			// Convert predictions to potential nuclei
			FloatIndexer indexerProb = matProb.createIndexer();
			FloatIndexer indexerRays = matRays.createIndexer();
			FloatIndexer indexerClassifications = matClassifications == null ? null : matClassifications.createIndexer();
			nuclei = createNuclei(indexerProb, indexerRays, indexerClassifications, request, scaleX, scaleY, mask);
			
			// Exclude anything that overlaps the right/bottom boundary of a region
			if (excludeOnBounds) {
				var iter = nuclei.iterator();
				while (iter.hasNext()) {
					var n = iter.next();
					var env = n.geometry.getEnvelopeInternal();
					if (env.getMaxX() >= request.getMaxX() || env.getMaxY() >= request.getMaxY())
						iter.remove();
				}
			}
		}

		return filterNuclei(nuclei);
	}
	
	/**
	 * Create a builder to customize detection parameters.
	 * This accepts either TensorFlow's savedmodel format (if TensorFlow is available) or alternatively a frozen 
	 * .pb file compatible with OpenCV's DNN module.
	 * @param modelPath path to the StarDist/TensorFlow model to use for prediction.
	 * @return
	 */
	public static Builder builder(String modelPath) {
		return new Builder(modelPath);
	}
	
	
	/**
	 * Create a builder to customize detection parameters, using a provided op for prediction.
	 * This provides a way to use an alternative machine learning library and model file, rather than the default 
	 * (OpenCV or TensorFlow).
	 * @param predictionOp the op to use for prediction
	 * @return
	 */
	public static Builder builder(ImageOp predictionOp) {
		return new Builder(predictionOp);		
	}
	
	
	/**
	 * Create a potential nucleus.
	 * @param indexerProb probability values
	 * @param indexerRays ray values
	 * @param indexerClass classification probabilities (optional)
	 * @param request region request, used to convert coordinates
	 * @param scaleX scaling to apply to x pixel index; normally 1.0, but may be 2.0 if passing downsampled output
	 * @param scaleY scaling to apply to y pixel index; normally 1.0, but may be 2.0 if passing downsampled output
	 * @param mask optional geometry mask, in the full image space
	 * @return list of potential nuclei, sorted in descending order of probability
	 */
	private List<PotentialNucleus> createNuclei(FloatIndexer indexerProb, FloatIndexer indexerRays, FloatIndexer indexerClass, RegionRequest request, double scaleX, double scaleY, Geometry mask) {
	    long[] sizes = indexerProb.sizes();
	    long[] inds = new long[3];
	    int h = (int)sizes[0];
	    int w = (int)sizes[1];
	    
	    int nRays = (int)indexerRays.size(2);
	    double[][] rays = sinCosAngles(nRays);
	    double[] raySine = rays[0];
	    double[] rayCosine = rays[1];
	    
	    int nClasses = indexerClass == null ? 0 : (int)indexerClass.size(2);

	    var nuclei = new ArrayList<PotentialNucleus>();
	    
	    var locator = mask == null ? null : new SimplePointInAreaLocator(mask);

	    var downsample = request.getDownsample();
	    var factory = GeometryTools.getDefaultFactory();
	    var precisionModel = factory.getPrecisionModel();
	    for (int y = 0; y < h; y++) {
	        inds[0] = y;
	        for (int x = 0; x < w; x++) {
	            inds[1] = x;
	            inds[2] = 0;
	            double prob = indexerProb.get(inds);
	            if (prob < threshold)
	                continue;
	            var coords = new ArrayList<Coordinate>();
	            Coordinate lastCoord = null;
	            for (int a = 0; a < nRays; a++) {
	                inds[2] = a;
	                double val = indexerRays.get(inds);
	                // We can get NaN
	                if (!Double.isFinite(val))
	                	continue;
	                // Python implementation imposes a minimum value
	                val = Math.max(1e-3, val);
	                // Create coordinate & add if it is distinct
	                double xx = precisionModel.makePrecise(request.getX() + (x * scaleX + val * rayCosine[a]) * downsample);
	                double yy = precisionModel.makePrecise(request.getY() + (y * scaleY + val * raySine[a]) * downsample);
	                var coord = new Coordinate(xx, yy);
	                if (!Objects.equals(coord, lastCoord))
	                	coords.add(coord);
	            }
	            // We need at least 3 for a reasonable nucleus
	            if (coords.size() < 3)
	            	continue;
	            else if (!coords.get(0).equals(coords.get(coords.size()-1)))
	            	coords.add(coords.get(0));
	            try {
		            var polygon = factory.createPolygon(coords.toArray(Coordinate[]::new));
		            if (locator == null || locator.locate(new Centroid(polygon).getCentroid()) != Location.EXTERIOR) {
		            	var geom = simplify(polygon);
		            	// Get classification, if available
		            	int classification = -1;
		            	if (indexerClass != null) {
		            		double maxProb = Double.NEGATIVE_INFINITY;
			            	for (int c = 0; c < nClasses; c++) {
			            		inds[2] = c;
			            		double probClass = indexerClass.get(inds);
			            		if (probClass > maxProb) {
				            		classification = c;
				            		maxProb = probClass;
			            		}
			            	}
		            	}
		            	if (classification != 0 || keepClassifiedBackground)
		            		nuclei.add(new PotentialNucleus(geom, prob, classification));
		            }
	            } catch (Exception e) {
	            	logger.warn("Error creating nucleus: " + e.getLocalizedMessage(), e);
	            }
	        }
	    }
	    return nuclei;
	}


	private List<PotentialNucleus> filterNuclei(List<PotentialNucleus> potentialNuclei) {
		
		// Sort in descending order of probability
		Collections.sort(potentialNuclei, Comparator.comparingDouble((PotentialNucleus n) -> n.getProbability()).reversed());
		
		// Create array of nuclei to keep & to skip
	    var nuclei = new LinkedHashSet<PotentialNucleus>();
	    var skippedNucleus = new HashSet<PotentialNucleus>();
	    int skipErrorCount = 0;
	    
	    // Create a spatial cache to find overlaps more quickly
	    // (Because of later tests, we don't need to update envelopes even though geometries may be modified)
	    Map<PotentialNucleus, Envelope> envelopes = new HashMap<>();
	    var tree = new STRtree();
	    for (var nuc : potentialNuclei) {
	    	var env = nuc.geometry.getEnvelopeInternal();
	    	envelopes.put(nuc, env);
	    	tree.insert(env, nuc);
	    }
	    
	    for (var nucleus : potentialNuclei) {
	        if (skippedNucleus.contains(nucleus))
	            continue;
	        
	        nuclei.add(nucleus);
        	var envelope = envelopes.get(nucleus);
        	
        	@SuppressWarnings("unchecked")
			var overlaps = (List<PotentialNucleus>)tree.query(envelope);
        	for (var nucleus2 : overlaps) {
        		if (nucleus2 == nucleus || skippedNucleus.contains(nucleus2) || nuclei.contains(nucleus2))
        			continue;
        		
            	// If we have an overlap, retain the higher-probability nucleus only (i.e. the one we met first)
        		// Try to refine other nuclei
	            try {
	            	var env = envelopes.get(nucleus2);
	                if (envelope.intersects(env) && nucleus.geometry.intersects(nucleus2.geometry)) {
	                	// Retain the nucleus only if it is not fragmented, or less than half its original area
	                    var difference = nucleus2.geometry.difference(nucleus.geometry);
	                    if (difference instanceof Polygon && difference.getArea() > nucleus2.fullArea / 2.0)
	                        nucleus2.geometry = difference;
	                    else {
	                    	skippedNucleus.add(nucleus2);
	                    }
	                }
	            } catch (Exception e) {
                	skippedNucleus.add(nucleus2);
	            	skipErrorCount++;
	            }

        	}
	    }
	    if (skipErrorCount > 0) {
	    	int skipCount = skippedNucleus.size();
	    	logger.warn("Skipped {} nucleus detection(s) due to error in resolving overlaps ({}% of all skipped)", 
	    			skipErrorCount, GeneralTools.formatNumber(skipErrorCount*100.0/skipCount, 1));
	    }
	    return new ArrayList<>(nuclei);
	}
	
	
	private static double[][] sinCosAngles(int n) {
	    double[][] angles = new double[2][n];
	    for (int i = 0; i < n; i++) {
	        double theta = 2 * Math.PI / n * i;
	        angles[0][i] = Math.sin(theta);
	        angles[1][i] = Math.cos(theta);
	    }
	    return angles;
	}
	
	
	private static class PotentialNucleus {
		
		private Geometry geometry;
	    private double fullArea;
	    private double probability;
	    private int classification;

	    PotentialNucleus(Geometry geom, double prob, int classification) {
	        this.geometry = geom;
	        this.probability = prob;
	        this.classification = classification;
	        this.fullArea = geom.getArea();
	    }

	    double getProbability() {
	        return probability;
	    };
	    
	    int getClassification() {
	    	return classification;
	    }
		
	}
	
}
