package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Encoders}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import MulticlassClassificationEvaluator._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}

import sys.process._

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // Import des données préparées parquet
    val df: DataFrame = spark
      .read
      .parquet("src/main/ressources/train/prepared_trainingset/")

    println(s"Nombre de lignes : ${df.count}")

    // Mise en place des différentes stages du futur pipeline

    // Récupération des mots (tokens) de la colonne "text"
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Suppression des stop words
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // Application de CountVectorizerModel sur la colonne tokens, qui est un array de mots
    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("tfidf")

    // Indexation des colonnes counyty2 et currency2
    val indexerCurrency2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep") // ceci permet de gérer les cas où des valeurs ne sont pas présentes dans le dataset de train mais présentes dans test

    val indexerCountry2 = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep") // ceci permet de gérer les cas où des valeurs ne sont pas présentes dans le dataset de train mais présentes dans test

    // Application de l'encodeur One Hot
    val oneHotEncoderCoun = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed"))
      .setOutputCols(Array("country_onehot"))

    val oneHotEncoderCurr = new OneHotEncoderEstimator()
      .setInputCols(Array("currency_indexed"))
      .setOutputCols(Array("currency_onehot"))

    // Assemblage tous les features en un unique vecteur
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot","currency_onehot"))
      .setOutputCol("features")

    // Déclaration de la regression logistique
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Mise en place du pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, indexerCurrency2, indexerCountry2, oneHotEncoderCoun, oneHotEncoderCurr, assembler, lr ))

    // Création des datasets de train et de test
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)

    // Entrainement du modèle sur le dataset de train
    val model = pipeline.fit(training)

    // Visualisation des résultats du train de la regression via un dataframe
    val dfTrain:DataFrame = model.transform(training)

    // Sauvegarde du pipeline sur ma machine
    model.write.overwrite().save("src/main/ressources/regression_logistique/")

    // Application du modèle sur le dataset de test
    val dfWithSimplePredictions:DataFrame = model.transform(test)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    // Création de la métrique f1score (par défaut evaluator calcule le f1)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val score = evaluator.evaluate(dfWithSimplePredictions)
    //println("f1 score = " + score)

    // Création de la grille d'hyperparamètres
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam,Array(10e-6, 10e-4,10e-2))
      .addGrid(cvModel.minDF,Array(65.0,75.0,85.0,95.0))
      .addGrid(lr.elasticNetParam, Array(1.0, 0.5, 0.01))
      .build()

    // Déclaration du trainValidationSplit
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    // Entrainement du modèle modelTuned sur le dataset d'entraînement
    val modelTuned = trainValidationSplit.fit(training)

    // Application du model au dataset de test
    val dfWithPredictions:DataFrame = modelTuned.transform(test)
    println("Résultat du meilleur modèle trouvé : ")
    dfWithPredictions.groupBy("final_status", "predictions").count.show()
    val scoreTuned = evaluator.evaluate(dfWithPredictions)


    // Sauvegarde du modèle tuned sur ma machine
    modelTuned.write.overwrite().save("src/main/ressources/regression_logistique_Tuned/")

    println(" ")
    println("f1 score = " + score)
    println("f1 score tuned = " + scoreTuned)
    println(" ")

    println("FIN")
  }
}



