# QuPath TensorFlow extension

Welcome to the TensorFlow extension for [QuPath](http://qupath.github.io)!

This adds support for loading some pre-trained TensorFlow models into QuPath 
using [TensorFlow for Java](http://github.com/tensorflow/java).

It is intended for the (at the time of writing) not-yet-released QuPath v0.3, 
and remains in a not-quite-complete state.
Its main use is in running [StarDist](https://qupath.readthedocs.io/en/0.2/docs/advanced/stardist.html) 
nucleus identification.

## Building

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

### GPU support

The default build process will use TensorFlow for the CPU.

As described at http://github.com/tensorflow/java there are alternative 
dependencies that include GPU support and/or use mkl.

To use any of these, add the platform to any of the building tasks above.
For example, to create a single GPU-friendly jar, use

```bash
gradlew clean shadowjar -P platform=gpu
```

The platforms available at the time of writing are `mkl`, `gpu`, `mkl-gpu`.


## Installing

The extension + its dependencies will all need to be available to QuPath inside 
Qupath's extensions folder.

The easiest way to install the jars is to simply drag them on top of QuPath 
when it's running.
You will then be prompted to ask whether you want to copy them to the 
appropriate folder.
