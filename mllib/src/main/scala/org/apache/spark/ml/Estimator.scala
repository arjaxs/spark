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

import scala.annotation.varargs

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.sql.Dataset

/**
 * :: DeveloperApi ::
 * Abstract class for estimators that fit models to data.
 */
@DeveloperApi
abstract class Estimator[M <: Model[M]] extends PipelineStage {

  /**
   * Fits a single model to the input data with optional parameters.
   *
   * @param dataset input dataset
   * @param firstParamPair the first param pair, overrides embedded params
   * @param otherParamPairs other param pairs.  These values override any specified in this
   *                        Estimator's embedded ParamMap.
   * @return fitted model
   */
  @Since("2.0.0")
  @varargs
  def fit(dataset: Dataset[_], firstParamPair: ParamPair[_], otherParamPairs: ParamPair[_]*): M = {
    MLEvents.withFitEvent(this, dataset) {
      fitImpl(dataset, firstParamPair, otherParamPairs: _*)
    }
  }

  /**
   * `fit()` handles events and then calls this method. Subclasses should override this
   * method to implement the actual fiting a model to the input data..
   */
  @Since("3.0.0")
  protected def fitImpl(
      dataset: Dataset[_], firstParamPair: ParamPair[_], otherParamPairs: ParamPair[_]*): M = {
    val map = new ParamMap()
      .put(firstParamPair)
      .put(otherParamPairs: _*)
    fitImpl(dataset, map)
  }

  /**
   * Fits a single model to the input data with provided parameter map.
   *
   * @param dataset input dataset
   * @param paramMap Parameter map.
   *                 These values override any specified in this Estimator's embedded ParamMap.
   * @return fitted model
   */
  @Since("2.0.0")
  def fit(dataset: Dataset[_], paramMap: ParamMap): M = {
    MLEvents.withFitEvent(this, dataset) {
      fitImpl(dataset, paramMap)
    }
  }

  /**
   * `fit()` handles events and then calls this method. Subclasses should override this
   * method to implement the actual fiting a model to the input data.
   */
  @Since("3.0.0")
  protected def fitImpl(dataset: Dataset[_], paramMap: ParamMap): M = {
    copy(paramMap).fitImpl(dataset)
  }

  /**
   * Fits a model to the input data.
   */
  @Since("2.0.0")
  def fit(dataset: Dataset[_]): M = MLEvents.withFitEvent(this, dataset) {
    fitImpl(dataset)
  }

  /**
   * `fit()` handles events and then calls this method. Subclasses should override this
   * method to implement the actual fiting a model to the input data.
   */
  @Since("3.0.0")
  protected def fitImpl(dataset: Dataset[_]): M = {
    // Keep this default body for backward compatibility.
    throw new UnsupportedOperationException("fitImpl is not implemented.")
  }

  /**
   * Fits multiple models to the input data with multiple sets of parameters.
   * The default implementation uses a for loop on each parameter map.
   * Subclasses could override this to optimize multi-model training.
   *
   * @param dataset input dataset
   * @param paramMaps An array of parameter maps.
   *                  These values override any specified in this Estimator's embedded ParamMap.
   * @return fitted models, matching the input parameter maps
   */
  @Since("2.0.0")
  def fit(dataset: Dataset[_], paramMaps: Array[ParamMap]): Seq[M] = {
    paramMaps.map { paramMap =>
      MLEvents.withFitEvent(this, dataset) {
        fitImpl(dataset, paramMap)
      }
    }
  }

  /**
   * `fit()` handles events and then calls this method. Subclasses should override this
   * method to implement the actual fiting a model to the input data..
   */
  @Since("3.0.0")
  protected def fitImpl(dataset: Dataset[_], paramMaps: Array[ParamMap]): Seq[M] = {
    paramMaps.map(fitImpl(dataset, _))
  }

  override def copy(extra: ParamMap): Estimator[M]
}
