/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.resnet

import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.dataset.ByteRecord
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

object Utils {
  case class TrainParams(
    folder: String = "./",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    optnet: Boolean = false,
    depth: Int = 20,
    classes: Int = 10,
    shortcutType: String = "A",
    batchSize: Int = 128,
    nepochs: Int = 165,
    learningRate: Double = 0.1,
    weightDecay: Double = 1e-4,
    momentum: Double = 0.9,
    movingRate: Double = 0.9,
    commPeriod: Int = 1,
    dampening: Double = 0.0,
    nesterov: Boolean = true)

  val trainParser = new OptionParser[TrainParams]("BigDL ResNet Example") {
    head("Train ResNet model on single node")
    opt[String]('f', "folder")
      .text("where you put your training files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Boolean]("optnet")
      .text("shared gradients and caches to reduce memory usage")
      .action((x, c) => c.copy(optnet = x))
    opt[Int]("depth")
      .text("depth of ResNet, 18 | 20 | 34 | 50 | 101 | 152 | 200")
      .action((x, c) => c.copy(depth = x))
    opt[Int]("classes")
      .text("classes of ResNet")
      .action((x, c) => c.copy(classes = x))
    opt[String]("shortcutType")
      .text("shortcutType of ResNet, A | B | C")
      .action((x, c) => c.copy(shortcutType = x))
    opt[Int]("batchSize")
      .text("batchSize of ResNet, 64 | 128 | 256 | ..")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]("nEpochs")
      .text("number of epochs of ResNet; default is 165")
      .action((x, c) => c.copy(nepochs = x))
    opt[Double]("learningRate")
      .text("initial learning rate of ResNet; default is 0.1")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("momentum")
      .text("momentum of ResNet; default is 0.9")
      .action((x, c) => c.copy(momentum = x))
    opt[Double]("movingRate")
      .text("movingRate of ResNet for EASGD; default is 0.9")
      .action((x, c) => c.copy(movingRate = x))
    opt[Int]("commPeriod")
      .text("commPeriod of ResNet for EASGD; default is 1")
      .action((x, c) => c.copy(commPeriod = x))
    opt[Double]("weightDecay")
      .text("weightDecay of ResNet; default is 1e-4")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]("dampening")
      .text("dampening of ResNet; default is 0.0")
      .action((x, c) => c.copy(dampening = x))
    opt[Boolean]("nesterov")
      .text("nesterov of ResNet; default is trye")
      .action((x, c) => c.copy(nesterov = x))
  }

  private[bigdl] def loadTrain(dataFile: String): Array[ByteRecord] = {
    val allFiles = Array(
      Paths.get(dataFile, "data_batch_1.bin"),
      Paths.get(dataFile, "data_batch_2.bin"),
      Paths.get(dataFile, "data_batch_3.bin"),
      Paths.get(dataFile, "data_batch_4.bin"),
      Paths.get(dataFile, "data_batch_5.bin")
    )

    val result = new ArrayBuffer[ByteRecord]()
    allFiles.foreach(load(_, result))
    result.toArray
  }

  private[bigdl] def loadTest(dataFile: String): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    val testFile = Paths.get(dataFile, "test_batch.bin")
    load(testFile, result)
    result.toArray
  }

  private[bigdl] def load(featureFile: Path, result : ArrayBuffer[ByteRecord]): Unit = {
    val rowNum = 32
    val colNum = 32
    val imageOffset = rowNum * colNum * 3 + 1
    val channelOffset = rowNum * colNum
    val bufferOffset = 8

    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(featureFile))
    val featureArray = featureBuffer.array()
    val featureCount = featureArray.length / (rowNum * colNum * 3 + 1)

    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum * 3 + bufferOffset))
      val byteBuffer = ByteBuffer.wrap(img)
      byteBuffer.putInt(rowNum)
      byteBuffer.putInt(colNum)

      val label = featureArray(i * imageOffset).toFloat
      var y = 0
      val start = i * imageOffset + 1
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img((x + y * colNum) * 3 + 2 + bufferOffset) =
            featureArray(start + x + y * colNum)
          img((x + y * colNum) * 3 + 1 + bufferOffset) =
            featureArray(start + x + y * colNum + channelOffset)
          img((x + y * colNum) * 3 + bufferOffset) =
            featureArray(start + x + y * colNum + 2 * channelOffset)
          x += 1
        }
        y += 1
      }
      result.append(ByteRecord(img, label + 1.0f))
      i += 1
    }
  }

}
