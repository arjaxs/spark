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

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Unstable
import org.apache.spark.ml.util.{MLReader, MLWriter}
import org.apache.spark.scheduler.SparkListenerEvent
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Event emitted by ML operations. Events are either fired before and/or
 * after each operation (the event should document this).
 */
@Unstable
sealed trait MLEvent extends SparkListenerEvent

/**
 * Event fired before `Transformer.transform`.
 */
@Unstable
case class TransformPreEvent(transformer: Transformer, input: Dataset[_]) extends MLEvent
/**
 * Event fired after `Transformer.transform`.
 */
@Unstable
case class TransformEvent(transformer: Transformer, output: Dataset[_]) extends MLEvent

/**
 * Event fired before `Estimator.fit`.
 */
@Unstable
case class FitPreEvent[M <: Model[M]](estimator: Estimator[M], dataset: Dataset[_]) extends MLEvent
/**
 * Event fired after `Estimator.fit`.
 */
@Unstable
case class FitEvent[M <: Model[M]](estimator: Estimator[M], model: M) extends MLEvent

/**
 * Event fired before `MLReader.load`.
 */
@Unstable
case class LoadInstancePreEvent[T](reader: MLReader[T], path: String) extends MLEvent
/**
 * Event fired after `MLReader.load`.
 */
@Unstable
case class LoadInstanceEvent[T](reader: MLReader[T], instance: T) extends MLEvent

/**
 * Event fired before `MLWriter.save`.
 */
@Unstable
case class SaveInstancePreEvent(writer: MLWriter, path: String) extends MLEvent
/**
 * Event fired after `MLWriter.save`.
 */
@Unstable
case class SaveInstanceEvent(writer: MLWriter, path: String) extends MLEvent


private[ml] object MLEvents {
  private lazy val listenerBus = SparkContext.getOrCreate().listenerBus

  def withFitEvent[M <: Model[M]](
      estimator: Estimator[M], dataset: Dataset[_])(func: => M): M = {
    listenerBus.post(FitPreEvent(estimator, dataset))
    val model: M = func
    listenerBus.post(FitEvent(estimator, model))
    model
  }

  def withTransformEvent(
      transformer: Transformer, input: Dataset[_])(func: => DataFrame): DataFrame = {
    listenerBus.post(TransformPreEvent(transformer, input))
    val output: DataFrame = func
    listenerBus.post(TransformEvent(transformer, output))
    output
  }

  def withLoadInstance[T](reader: MLReader[T], path: String)(func: => T): T = {
    listenerBus.post(LoadInstancePreEvent(reader, path))
    val instance: T = func
    listenerBus.post(LoadInstanceEvent(reader, instance))
    instance
  }

  def withSaveInstance(writer: MLWriter, path: String)(func: => Unit): Unit = {
    listenerBus.post(SaveInstancePreEvent(writer, path))
    func
    listenerBus.post(SaveInstanceEvent(writer, path))
  }
}
