package br.gsj.spark.kmeans

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import vegas._
import java.io.PrintWriter
import java.io.File
import br.gsj.utils.ImageUtils

/**
 * plot the optimal number of clusters K based on the K-means cost function for a range of K
 */
object ImageOptimalKComputation {
    case class Image(data: Byte)

  
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder().appName("mriClass").master("local[*]").getOrCreate()
    val image_path = "src/main/resources/m1.jpg"
    
//    val image_df = spark.read.format("image").load(mri_healthy_brain_image).select(col("image.*"))
//    image_df.show
//    image_df.printSchema
    
    val final_image_df = ImageUtils.decodeImageDataFrame(spark, image_path)
    
    final_image_df.printSchema

   
    
//    
    val features_col = Array("r","g","b")
    val vector_assembler = new VectorAssembler()
    .setInputCols(features_col)
    .setOutputCol("features")
//    
    val features_df = vector_assembler.transform(final_image_df).select("features")
  
    val cost = new Array[Double](21)
//    
    for(k<-2 to 20){
      val kmeans = new KMeans().setK(k).setSeed(1L).setFeaturesCol("features")
      val model =  kmeans.fit(features_df.sample(false,0.1))
      cost(k) = model.computeCost(features_df)
    }
//    
    var plotSeq = 
        cost.zipWithIndex.map{case (l,i)=> collection.mutable.Map("clusters" -> i, "cost" -> l)}
    .map(p => p.retain((k,v) => v != 0)).map(f => Map(f.toSeq:_*))
        
    val writer = new PrintWriter(new File("src/main/resources/kmeans/optimalCost.html"))

    val plot = Vegas("Kmeans Cost").withData(plotSeq).
      encodeX("clusters", Nom).
      encodeY("cost", Quant).
      mark(Line)

     writer.write(plot.html.pageHTML("plot"))
     writer.close()
// 
  
    
  }
  
}