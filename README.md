# QuPath TensorFlow extension

Welcome to the TensorFlow extension for [QuPath](http://qupath.github.io)!

This adds support for loading some pre-trained TensorFlow models into QuPath 
using [TensorFlow for Java](http://github.com/tensorflow/java).

It is intended for the (at the time of writing) not-yet-released QuPath v0.3, 
and remains in a not-quite-complete state.

Its previous main use was to run [StarDist](https://qupath.readthedocs.io/en/0.2/docs/advanced/stardist.html) 
nucleus identification, although the new [QuPath StarDist extension](https://github.com/qupath/qupath-extension-stardist) does not require that TensorFlow is available.

> **Important!** TensorFlow Java does not currently support Mac computers with Apple silicon.


## Building

There is no pre-built version of the TensorFlow extension at this time, because most people shouldn't need it and there are lots of different permutations of dependencies that might be required for different platforms.

For that reason, it needs to be built from source.
If you want to match to a specific version, you can download the source from the [releases](https://github.com/qupath/qupath-extension-tensorflow/releases) page.


### Extension + dependencies separately

You can build the extension with

```bash
gradlew clean build copyDependencies
```

The output will be under `build/libs`.

* `clean` removes anything old
* `build` builds the QuPath extension as a *.jar* file and adds it to `libs`
* `copyDependencies` copies the TensorFlow dependencies to the `libs` folder

### Extension + dependencies together

Alternatively, you can create a single *.jar* file that contains both the 
extension and all its dependencies with

```bash
gradlew clean shadowjar
```

### Alternative platforms

The default build process will use TensorFlow for the CPU.

As described at http://github.com/tensorflow/java there are alternative 
dependencies that include GPU support and/or use mkl.

To use any of these, add the platform to any of the building tasks above.
For example, to create a single GPU-friendly jar, use

```bash
gradlew clean build copyDependencies -P platform=gpu
```

The platforms available at the time of writing are `mkl`, `gpu`, `mkl-gpu`.

> Not all options are available for all operating systems.
> For example, GPU support is not available with macOS.


### GPU support

When using `platform=gpu`, you will need
* an NVIDIA GPU
* CUDA and cuDNN

Installation may be simplified if you include 

```bash
gradlew clean build copyDependencies -P platform=gpu -Pcuda-redist
```

to download the required CUDA files via JavaCPP. 

Before using this option, please check https://github.com/bytedeco/javacpp-presets/tree/master/cuda for 
the terms of license agreements for NVIDIA software included in the archives.

> **Warning!** At the time of writing, the CUDA version used with TensorFlow Java differs from that  
> used with OpenCV via JavaCPP. This is likely to cause problems if trying to use both.


## Installing

The extension + its dependencies will all need to be available to QuPath inside 
QuPath's extensions folder.

The easiest way to install the jars is to simply drag them on top of QuPath 
when it's running.
You will then be prompted to ask whether you want to copy them to the 
appropriate folder.
