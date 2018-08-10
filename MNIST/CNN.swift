// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlow

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>, labels: Tensor<Int32>) {
  print("Reading data.")
  let imageData = try! Data(contentsOf: URL(fileURLWithPath: imagesFile)).dropFirst(16)
  let labelData = try! Data(contentsOf: URL(fileURLWithPath: labelsFile)).dropFirst(8)
  let images = imageData.map { Float($0) }
  let labels = labelData.map { Int32($0) }
  let rowCount = Int32(labels.count)
  let columnCount = Int32(images.count) / rowCount

  print("Constructing data tensors.")
  let imagesTensor = Tensor(shape: [rowCount, columnCount], scalars: images) / 255
  let labelsTensor = Tensor(labels)
  return (imagesTensor.toAccelerator(), labelsTensor.toAccelerator())
}

/// Parameters of an MNIST classifier.
struct MNISTParameters : ParameterAggregate {
  var w1 = Tensor<Float>(randomNormal: [5, 5, 1, 32], stddev: 0.1)
  var w2 = Tensor<Float>(randomNormal: [5, 5, 32, 64], stddev: 0.1)
  var wDense1 = Tensor<Float>(randomNormal: [Int32(7 * 7 * 64), 1024], stddev: 0.1)
  var bDense1 = Tensor<Float>(shape: [1024], repeating: 0.1)
  var wDense2 = Tensor<Float>(randomNormal: [1024, 10], stddev: 0.1)
  var bDense2 = Tensor<Float>(shape: [10], repeating: 0.1)

  // var w1 = Tensor<Float>(shape: [5, 5, 1, 32], repeating: 0.1)
  // var w2 = Tensor<Float>(shape: [5, 5, 32, 64], repeating: 0.1)
  // var wDense1 = Tensor<Float>(shape: [Int32(7 * 7 * 64), 1024], repeating: 0.1)
  // var bDense1 = Tensor<Float>(shape: [1024], repeating: 0.1)
  // var wDense2 = Tensor<Float>(shape: [1024, 10], repeating: 0.1)
  // var bDense2 = Tensor<Float>(shape: [10], repeating: 0.1)

  static var allKeyPaths: [WritableKeyPath<MNISTParameters, Tensor<Float>>] {
    return [
      \MNISTParameters.w1,
      \MNISTParameters.w2,
      \MNISTParameters.wDense1,
      \MNISTParameters.bDense1,
      \MNISTParameters.wDense2,
      \MNISTParameters.bDense2,
    ]
  }
}

struct AdamOptimizer {
  var learningRate: Float
  var beta1: Float
  var beta2: Float
  var epsilon: Float

  init(learningRate: Float = 0.001, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8) {
    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
  }

  var step: Float = 0
  var movingAverages: MNISTParameters? = nil
  var movingSquaredAverages: MNISTParameters? = nil

  mutating func fitParameters(
    _ parameters: inout MNISTParameters,
    withGradients gradients: MNISTParameters
  ) {
    func initializeWithZerosIfNeeded(_ x: MNISTParameters?) -> MNISTParameters {
      return x ?? MNISTParameters(
        w1: Tensor(0).broadcast(like: parameters.w1),
        w2: Tensor(0).broadcast(like: parameters.w2),
        wDense1: Tensor(0).broadcast(like: parameters.wDense1),
        bDense1: Tensor(0).broadcast(like: parameters.bDense1),
        wDense2: Tensor(0).broadcast(like: parameters.wDense2),
        bDense2: Tensor(0).broadcast(like: parameters.bDense2)
      )
    }
    var movingAverages = initializeWithZerosIfNeeded(self.movingAverages)
    var movingSquaredAverages = initializeWithZerosIfNeeded(self.movingSquaredAverages)

    step += 1

    for kp in MNISTParameters.allKeyPaths {
      movingAverages[keyPath: kp] =
        movingAverages[keyPath: kp] * beta1 + (1 - beta1) * gradients[keyPath: kp]
      movingSquaredAverages[keyPath: kp] =
        movingAverages[keyPath: kp] * beta2 + (1 - beta2) * gradients[keyPath: kp] * gradients[keyPath: kp]

      // movingAverages[keyPath: kp] *= beta1 + (1 - beta1) * gradients[keyPath: kp]
      // movingSquaredAverages[keyPath: kp] *= beta2 + (1 - beta2) * gradients[keyPath: kp] * gradients[keyPath: kp]
      // movingAverages[keyPath: kp] *= beta1
      // movingAverages[keyPath: kp] += (1 - beta1) * gradients[keyPath: kp]
      // movingSquaredAverages[keyPath: kp] *= beta2
      // movingSquaredAverages[keyPath: kp] += (1 - beta2) * gradients[keyPath: kp] * gradients[keyPath: kp]

      let denominator = sqrt(movingSquaredAverages[keyPath: kp]) + epsilon
      let biasCorrection1 = 1 - pow(beta1, step)
      let biasCorrection2 = 1 - pow(beta2, step)
      let stepSize = learningRate * sqrt(biasCorrection2) / biasCorrection1
      parameters[keyPath: kp] -= stepSize * movingAverages[keyPath: kp] / denominator

      /*
      parameters[keyPath: kp] =
        parameters[keyPath: kp] - learningRate *
        movingAverages[keyPath: kp] / (sqrt(movingSquaredAverages[keyPath: kp]) + epsilon)
      */
    }
    self.movingAverages = movingAverages
    self.movingSquaredAverages = movingSquaredAverages
  }
}

/// Train a MNIST classifier for the specified number of epochs.
// func train(epochCount: Int32) {
func train(_ parameters: inout MNISTParameters, epochCount: Int32) {
// func train(_ parameters: inout MNISTParameters) { let epochCount: Int32 = 20
// func train() { let epochCount: Int32 = 20
  // Get script path. This is necessary for MNIST.swift to work when
  // invoked from any directory.
  let currentScriptPath = URL(fileURLWithPath: #file)
  let scriptDirectory = currentScriptPath.deletingLastPathComponent()

  // Get training data.
  let imagesFile = scriptDirectory.appendingPathComponent("train-images-idx3-ubyte").path
  let labelsFile = scriptDirectory.appendingPathComponent("train-labels-idx1-ubyte").path
  let (rawImages, numericLabels) = readMNIST(imagesFile: imagesFile, labelsFile: labelsFile)
  let images = rawImages.reshaped(toShape: [-1, 28, 28, 1])
  let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)
  let batchSize = Float(images.shape[0])

  // Hyper-parameters.
  // let minibatchSize: Int32 = 100
  let minibatchSize: Int32 = 50
  // let learningRate: Float = 0.01
  let learningRate: Float = 0.001
  var loss = Float.infinity

  // Optimizer states.
  /*
  let w1ExpAvg = Tensor(0).broadcast(like: w1)
  let w2ExpAvg = Tensor(0).broadcast(like: w2)
  let wDense1ExpAvg = Tensor(0).broadcast(like: wDense1)
  let wDense2ExpAvg = Tensor(0).broadcast(like: wDense2)
  let bDense1ExpAvg = Tensor(0).broadcast(like: bDense1)
  let bDense2ExpAvg = Tensor(0).broadcast(like: bDense2)
  */

  // Training loop.
  print("Begin training for \(epochCount) epochs.")

  var optimizer = AdamOptimizer()

  func minibatch<Scalar>(_ x: Tensor<Scalar>, index: Int32) -> Tensor<Scalar> {
    let start = index * minibatchSize
    return x[start..<start+minibatchSize]
  }

  // TODO: Replace repeat-while loop with for-in loop after SR-8376 is fixed.
  // for _ in 0..<epochCount {
  var epoch: Int32 = 0
  repeat {
    // Store number of correct/total guesses, used to print accuracy.
    var correctGuesses = 0
    var totalGuesses = 0
    // TODO: Randomly sample minibatches using TensorFlow dataset APIs.
    let iterationCount = Int32(batchSize) / minibatchSize
    // for i in 0..<iterationCount {
    var i: Int32 = 0
    repeat {
      // let images = minibatch(images, index: i)
      // let numericLabels = minibatch(numericLabels, index: i)
      // let labels = minibatch(labels, index: i)
      let start = i * minibatchSize
      let images = images[start..<start+minibatchSize]
      let numericLabels = numericLabels[start..<start+minibatchSize]
      let labels = labels[start..<start+minibatchSize]

      // Forward pass.
      let conv1 = images.convolved2D(withFilter: parameters.w1, strides: (1, 1, 1, 1), padding: .same)
      let h1 = relu(conv1)
      let pool1 = h1.maxPooled(kernelSize: (1, 2, 2, 1), strides: (1, 2, 2, 1), padding: .same)

      let conv2 = pool1.convolved2D(withFilter: parameters.w2, strides: (1, 1, 1, 1), padding: .same)
      let h2 = relu(conv2)
      let pool2 = h2.maxPooled(kernelSize: (1, 2, 2, 1), strides: (1, 2, 2, 1), padding: .same)

      let flat = pool2.reshaped(toShape: Tensor<Int32>([-1, Int32(7 * 7 * 64)]))
      // let flat = pool2.reshaped(toShape: Tensor<Int32>([-1, 3136]))
      let dense1 = matmul(flat, parameters.wDense1) + parameters.bDense1
      let h3 = relu(dense1)

      // Note: dropout is unimplemented and missing.

      let dense2 = matmul(h3, parameters.wDense2) + parameters.bDense2
      let h4 = relu(dense2)

      // print(h4)
      // let predictions = sigmoid(dense2)
      let predictions = sigmoid(dense2)
      // let (loss, gradients) = Raw.softmaxCrossEntropyWithLogits(features: dense2, labels: labels)
      let (loss, dpredictions) = Raw.sparseSoftmaxCrossEntropyWithLogits(features: dense2, labels: numericLabels)

      // print("predictions", predictions)
      // print("loss", loss)
      // print("gradients", gradients)

      // Backward pass. This will soon be replaced by automatic
      // differentiation.
      // let dpredictions = predictions - labels
      // let dh4 = #adjoint(relu)(dense2, originalValue: h4, seed: dpredictions)
      let dh4 = Raw.reluGrad(gradients: dpredictions, features: dense2)
      let dbDense2 = dpredictions.sum(squeezingAxes: 0)
      let (ddense2_lhs, dwDense2) = #adjoint(matmul)(
        h3, parameters.wDense2, originalValue: dense2, seed: dh4
      )

      // let dh3 = #adjoint(relu)(dense1, originalValue: h3, seed: ddense2_lhs)
      let dh3 = Raw.reluGrad(gradients: ddense2_lhs, features: dense1)
      let dbDense1 = dh3.sum(squeezingAxes: 0)
      let (ddense1_lhs, dwDense1) = #adjoint(matmul)(
        flat, parameters.wDense1, originalValue: dense1, seed: dh3
      )
      let dflat = ddense1_lhs.reshaped(like: pool2)

      let dpool2 = #adjoint(Tensor.maxPooled)(h2)(
        kernelSize: (1, 2, 2, 1),
        strides: (1, 2, 2, 1),
        padding: .same,
        originalValue: pool2,
        seed: dflat
      )
      // let dh2 = #adjoint(relu)(conv2, originalValue: h2, seed: dpool2)
      let dh2 = Raw.reluGrad(gradients: dpool2, features: conv2)
      let (dconv2_in, dw2) = #adjoint(Tensor.convolved2D)(pool1)(filter: parameters.w2, strides: (1, 1, 1, 1), padding: .same, originalValue: conv2, seed: dh2)

      let dpool1 = #adjoint(Tensor.maxPooled)(h1)(kernelSize: (1, 2, 2, 1), strides: (1, 2, 2, 1), padding: .same, originalValue: pool1, seed: dconv2_in)
      // let dh1 = #adjoint(relu)(conv1, originalValue: h1, seed: dpool1)
      let dh1 = Raw.reluGrad(gradients: dpool1, features: conv1)
      let (_, dw1) = #adjoint(Tensor.convolved2D)(images)(filter: parameters.w1, strides: (1, 1, 1, 1), padding: .same, originalValue: conv1, seed: dh1)
      let gradients = MNISTParameters(
        w1: dw1, w2: dw2, wDense1: dwDense1, bDense1: dbDense1, wDense2: dwDense2, bDense2: dbDense2
      )

      // Update parameters.
      /*
      parameters.update(withGradients: gradients) { param, grad in
        param -= grad * learningRate
      }
      */
      optimizer.fitParameters(&parameters, withGradients: gradients)

      /*
      w1 -= learningRate * dw1
      w2 -= learningRate * dw2
      wDense1 -= learningRate * dwDense1
      bDense1 -= learningRate * dbDense1
      wDense2 -= learningRate * dwDense2
      bDense2 -= learningRate * dbDense2
      */

      // Loss formula from:
      // https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
      // let part1 = max(predictions, 0) - predictions * labels
      // let part2 = log(1 + exp(-abs(predictions)))
      // loss = ((part1 + part2).sum(squeezingAxes: 0, 1) / batchSize).scalarized()

      // Update number of correct/total guesses.
      let predictedLabels = predictions.argmax(squeezingAxis: 1)
      // print(predictedLabels)
      // print(numericLabels)
      let correctPredictions = predictions.argmax(squeezingAxis: 1).elementsEqual(numericLabels)
      correctGuesses += Int(Tensor<Int32>(correctPredictions).sum())
      totalGuesses += Int(minibatchSize)
      print("""
            Accuracy: \(correctGuesses)/\(totalGuesses) \
            (\(Float(correctGuesses) / Float(totalGuesses)))
            """)
        i += 1
      } while i < iterationCount
    // }
  // }
    epoch += 1
  } while epoch < epochCount
}

var parameters = MNISTParameters()
// Start training.
// train(epochCount: 20)
train(&parameters, epochCount: 20)
// train(&parameters)
// train()
