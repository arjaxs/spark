/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml

import scala.collection.mutable
import scala.concurrent.duration._
import scala.language.postfixOps

import org.apache.hadoop.fs.Path
import org.mockito.Matchers.{any, eq => meq}
import org.mockito.Mockito.when
import org.scalatest.{BeforeAndAfterEach, PrivateMethodTester}
import org.scalatest.concurrent.Eventually
import org.scalatest.mockito.MockitoSugar.mock

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.scheduler.{SparkListener, SparkListenerEvent}
import org.apache.spark.sql._


class MLEventsSuite
    extends SparkFunSuite
    with BeforeAndAfterEach
    with MLlibTestSparkContext
    with Eventually {

  private val events = mutable.ArrayBuffer.empty[MLEvent]
  private val listener: SparkListener = new SparkListener {
    override def onOtherEvent(event: SparkListenerEvent): Unit = event match {
      case e: FitStart[_] => events.append(e)
      case e: FitEnd[_] => events.append(e)
      case e: TransformStart => events.append(e)
      case e: TransformEnd => events.append(e)
      case e: SaveInstanceStart => events.append(e)
      case e: SaveInstanceEnd => events.append(e)
      case _ =>
    }
  }

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark.sparkContext.addSparkListener(listener)
  }

  override def afterEach(): Unit = {
    try {
      events.clear()
    } finally {
      super.afterEach()
    }
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.sparkContext.removeSparkListener(listener)
      }
    } finally {
      super.afterAll()
    }
  }

  abstract class MyModel extends Model[MyModel]

  test("pipeline fit events") {
    val estimator1 = mock[Estimator[MyModel]]
    val model1 = mock[MyModel]
    val transformer1 = mock[Transformer]
    val estimator2 = mock[Estimator[MyModel]]
    val model2 = mock[MyModel]

    when(estimator1.copy(any[ParamMap])).thenReturn(estimator1)
    when(model1.copy(any[ParamMap])).thenReturn(model1)
    when(transformer1.copy(any[ParamMap])).thenReturn(transformer1)
    when(estimator2.copy(any[ParamMap])).thenReturn(estimator2)
    when(model2.copy(any[ParamMap])).thenReturn(model2)

    val dataset0 = mock[DataFrame]
    val dataset1 = mock[DataFrame]
    val dataset2 = mock[DataFrame]
    val dataset3 = mock[DataFrame]
    val dataset4 = mock[DataFrame]

    when(dataset0.toDF).thenReturn(dataset0)
    when(dataset1.toDF).thenReturn(dataset1)
    when(dataset2.toDF).thenReturn(dataset2)
    when(dataset3.toDF).thenReturn(dataset3)
    when(dataset4.toDF).thenReturn(dataset4)

    when(estimator1.fit(meq(dataset0))).thenReturn(model1)
    when(model1.transform(meq(dataset0))).thenReturn(dataset1)
    when(model1.parent).thenReturn(estimator1)
    when(transformer1.transform(meq(dataset1))).thenReturn(dataset2)
    when(estimator2.fit(meq(dataset2))).thenReturn(model2)

    assert(events.isEmpty)
    val pipeline = new Pipeline()
      .setStages(Array(estimator1, transformer1, estimator2))
    val pipelineModel = pipeline.fit(dataset0)
    val expected =
      FitStart(pipeline, dataset0) ::
      FitStart(estimator1, dataset0) ::
      FitEnd(estimator1, model1) ::
      TransformStart(model1, dataset0) ::
      TransformEnd(model1, dataset1) ::
      TransformStart(transformer1, dataset1) ::
      TransformEnd(transformer1, dataset2) ::
      FitStart(estimator2, dataset2) ::
      FitEnd(estimator2, model2) ::
      FitEnd(pipeline, pipelineModel) :: Nil
    eventually(timeout(10 seconds), interval(1 second)) {
      assert(expected === events)
    }
  }

  test("pipeline model transform events") {
    val dataset = mock[DataFrame]
    when(dataset.toDF).thenReturn(dataset)
    val transform1 = mock[Transformer]
    val model = mock[MyModel]
    val transform2 = mock[Transformer]
    val stages = Array(transform1, model, transform2)
    val newPipelineModel = new PipelineModel("pipeline0", stages)
    val output = newPipelineModel.transform(dataset)

    val expected =
      TransformStart(newPipelineModel, dataset) ::
      TransformEnd(newPipelineModel, output) :: Nil
    eventually(timeout(10 seconds), interval(1 second)) {
      assert(expected === events)
    }
  }

  test("pipeline read/write events") {
    def getInstance(w: MLWriter): AnyRef =
      w.getClass.getDeclaredMethod("instance").invoke(w)

    withTempDir { dir =>
      val path = new Path(dir.getCanonicalPath, "pipeline").toUri.toString
      val writableStage = new WritableStage("writableStage")
      val newPipeline = new Pipeline().setStages(Array(writableStage))
      val pipelineWriter = newPipeline.write
      pipelineWriter.save(path)

      eventually(timeout(10 seconds), interval(1 second)) {
        events.foreach {
          case e: SaveInstanceStart =>
            assert(getInstance(e.writer).asInstanceOf[Pipeline].uid === newPipeline.uid)
          case e: SaveInstanceEnd =>
            assert(getInstance(e.writer).asInstanceOf[Pipeline].uid === newPipeline.uid)
          case e => fail(s"Unexpected event thrown: $e")
        }
      }
    }
  }

  test("pipeline model read/write events") {
    def getInstance(w: MLWriter): AnyRef =
      w.getClass.getDeclaredMethod("instance").invoke(w)

    withTempDir { dir =>
      val path = new Path(dir.getCanonicalPath, "pipeline").toUri.toString
      val writableStage = new WritableStage("writableStage")
      val pipelineModel =
        new PipelineModel("pipeline_89329329", Array(writableStage.asInstanceOf[Transformer]))
      val pipelineWriter = pipelineModel.write
      pipelineWriter.save(path)

      eventually(timeout(10 seconds), interval(1 second)) {
        events.foreach {
          case e: SaveInstanceStart =>
            assert(getInstance(e.writer).asInstanceOf[PipelineModel].uid === pipelineModel.uid)
          case e: SaveInstanceEnd =>
            assert(getInstance(e.writer).asInstanceOf[PipelineModel].uid === pipelineModel.uid)
          case e => fail(s"Unexpected events were thrown: $e")
        }
      }
    }
  }
}
