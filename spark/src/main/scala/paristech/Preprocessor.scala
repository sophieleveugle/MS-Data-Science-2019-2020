package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.expr
import org.apache.spark.sql.types.DateType

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/ressources/train/train_clean.csv")


    // Statistiques descriptives
    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")
    //df.show()
    //df.printSchema()

    // Changement de type de STRING vers INT de colonnes éligibles
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    //dfCasted.printSchema()

    //dfCasted
    //  .select("goal", "backers_count", "final_status")
    //  .describe()
    //  .show

    // Suppression de la colonne disable_communication
    val df2: DataFrame = dfCasted
      .drop("disable_communication")

    val dfNoFutur: DataFrame = df2
      .drop("backers_count", "state_changed_at")

    // Conversion des dates (int) en timestamp
    val dfConversion: DataFrame = dfNoFutur
      .withColumn("deadline_ts", from_unixtime($"deadline"))
      .withColumn("created_at_ts", from_unixtime($"created_at"))
      .withColumn("launched_at_ts", from_unixtime($"launched_at"))
      .drop("deadline", "created_at","launched_at")

    //newDF.show()

    // Nettoyage des colonnes country et currency
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfConversion
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")


    // Calcul de nouvelles colonnes : days_campaign et hours_prepa
    // Conversion en minuscules des colonnes desc, keywords et name
    // Nettoyage de la colonne name
    val dfDate:DataFrame = dfCountry
      .withColumn("days_campaign", ceil((unix_timestamp($"deadline_ts") - unix_timestamp($"launched_at_ts"))/ (24D * 3600D)))
      .withColumn("hours_prepa", (unix_timestamp($"launched_at_ts") - unix_timestamp($"created_at_ts"))*24/ (24D * 3600D))
      .withColumn("hours_prepa", round($"hours_prepa",3))
      .withColumn("desc", lower(col("desc")))
      .withColumn("keywords", lower(col("keywords")))
      .withColumn("name", lower(col("name")))
      .withColumn("name", regexp_replace(col("name") , "\"", "" ))
      .withColumn("text", concat_ws(" ", $"name",$"desc",$"keywords")) //$"name"+" "+$"desc"+" "+$"keywords"))

    //dfDate
    //  .select($"name")
    //  .show(false)

    // Remplacement des valeurs NULL
    val dfNull:DataFrame = dfDate
      .na.fill(-1,Seq("days_campaign"))
      .na.fill(-1,Seq("hours_prepa"))
      .na.fill(-1,Seq("goal"))
      .na.fill("unknown",Seq("country2"))
      .na.fill("unknown",Seq("currency2"))

    // Export du dataframe en parquet
    dfNull.write.mode("overwrite").parquet("src/main/ressources/preprocessed")

    println(" ")
    println("FIN")
  }
}
