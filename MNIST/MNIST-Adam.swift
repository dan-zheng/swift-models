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
    var w1 = Tensor<Float>(randomNormal: [784, 30])
    var w2 = Tensor<Float>(randomNormal: [30, 10])
    var b1 = Tensor<Float>(zeros: [1, 30])
    var b2 = Tensor<Float>(zeros: [1, 10])

    static var allKeyPaths: [WritableKeyPath<MNISTParameters, Tensor<Float>>] {
      return [
        \MNISTParameters.w1,
        \MNISTParameters.w2,
        \MNISTParameters.b1,
        \MNISTParameters.b2,
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
        // w1: Tensor(0).broadcast(like: parameters.w1),
        // w2: Tensor(0).broadcast(like: parameters.w2),
        // b1: Tensor(0).broadcast(like: parameters.b1),
        // b2: Tensor(0).broadcast(like: parameters.b2)
        w1: Tensor(0),
        w2: Tensor(0),
        b1: Tensor(0),
        b2: Tensor(0)
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
      // movingAverages[keyPath: kp] *= beta1
      // movingAverages[keyPath: kp] += (1 - beta1) * gradients[keyPath: kp]
      // movingSquaredAverages[keyPath: kp] *= beta2
      // movingSquaredAverages[keyPath: kp] += (1 - beta2) * gradients[keyPath: kp] * gradients[keyPath: kp]

      /*
      // let denominator = sqrt(movingSquaredAverages[keyPath: kp]) + epsilon
      let denominator = sqrt(movingSquaredAverages[keyPath: kp]) + epsilon
      let biasCorrection1 = 1 - pow(beta1, step)
      let biasCorrection2 = 1 - pow(beta2, step)
      let stepSize = learningRate * sqrt(biasCorrection2) / biasCorrection1
      print(stepSize * movingAverages[keyPath: kp] / denominator)
      parameters[keyPath: kp] -= stepSize * movingAverages[keyPath: kp] / denominator
      */

      // let denominator = sqrt(movingSquaredAverages[keyPath: kp]) + epsilon
      let biasCorrected1 = movingAverages[keyPath: kp] / (1 - pow(beta1, step))
      let biasCorrected2 = movingSquaredAverages[keyPath: kp] / (1 - pow(beta2, step))
      // let stepSize = learningRate * sqrt(biasCorrection2) / biasCorrection1
      let alpha = learningRate * sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step))
      parameters[keyPath: kp] -= alpha * biasCorrected1 / (sqrt(biasCorrected2 + epsilon))
    }
    // print(movingAverages)
    self.movingAverages = movingAverages
    self.movingSquaredAverages = movingSquaredAverages
  }
}

/// Train a MNIST classifier for the specified number of epochs.
func train(_ parameters: inout MNISTParameters, epochCount: Int32) {
    // Get script path. This is necessary for MNIST.swift to work when invoked
    // from any directory.
    let currentScriptPath = URL(fileURLWithPath: #file)
    let scriptDirectory = currentScriptPath.deletingLastPathComponent()

    // Get training data.
    let imagesFile = scriptDirectory.appendingPathComponent("train-images-idx3-ubyte").path
    let labelsFile = scriptDirectory.appendingPathComponent("train-labels-idx1-ubyte").path
    let (images, numericLabels) = readMNIST(imagesFile: imagesFile, labelsFile: labelsFile)
    let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)
    let batchSize = Float(images.shape[0])

    // Hyper-parameters.
    let minibatchSize: Int32 = 10
    let learningRate: Float = 0.2
    var loss = Float.infinity

    // Training loop.
    print("Begin training for \(epochCount) epochs.")

    var optimizer = AdamOptimizer()

    func minibatch<Scalar>(_ x: Tensor<Scalar>, index: Int32) -> Tensor<Scalar> {
      let start = index * minibatchSize
      return x[start..<start+minibatchSize]
    }

    for _ in 0...epochCount {
        // Store number of correct/total guesses, used to print accuracy.
        var correctGuesses = 0
        var totalGuesses = 0

        // TODO: Randomly sample minibatches using TensorFlow dataset APIs.
        let iterationCount = Int32(batchSize) / minibatchSize
        for i in 0..<iterationCount {
            let images = minibatch(images, index: i)
            let numericLabels = minibatch(numericLabels, index: i)
            let labels = minibatch(labels, index: i)

            // Forward pass.
            let z1 = images • parameters.w1 + parameters.b1
            let h1 = sigmoid(z1)
            let z2 = h1 • parameters.w2 + parameters.b2
            let predictions = sigmoid(z2)

            // Backward pass. This will soon be replaced by automatic
            // differentiation.
            let dz2 = predictions - labels
            let dw2 = h1.transposed() • dz2
            let db2 = dz2.sum(squeezingAxes: 0)
            let dz1 = matmul(dz2, parameters.w2.transposed()) * h1 * (1 - h1)
            let dw1 = images.transposed() • dz1
            let db1 = dz1.sum(squeezingAxes: 0)
            let gradients = MNISTParameters(w1: dw1, w2: dw2, b1: db1, b2: db2)

            // Update parameters.
            optimizer.fitParameters(&parameters, withGradients: gradients)

            // Calculate the sigmoid-based cross-entropy loss.
            // TODO: Use softmax-based cross-entropy loss instead. Sigmoid
            // cross-entropy loss treats class labels as independent, which is
            // unnecessary for single-label classification tasks like MNIST.
            // Sigmoid cross-entropy formula from:
            // https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

            // let part1 = max(predictions, 0) - predictions * labels
            // let part2 = log(1 + exp(-abs(predictions)))
            // loss = ((part1 + part2).sum(squeezingAxes: 0, 1) / batchSize).scalarized()

            // Update number of correct/total guesses.
            let correctPredictions = predictions.argmax(squeezingAxis: 1).elementsEqual(numericLabels)
            correctGuesses += Int(Tensor<Int32>(correctPredictions).sum())
            totalGuesses += Int(minibatchSize)
        }
        print("""
              Accuracy: \(correctGuesses)/\(totalGuesses) \
              (\(Float(correctGuesses) / Float(totalGuesses)))
              """)
    }
}

var parameters = MNISTParameters()
// Start training.
train(&parameters, epochCount: 20)
