package br.gsj.spark.kmeans

import org.apache.spark.sql.SparkSession
import java.io.ByteArrayInputStream
import java.io.File

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.awt.image.MemoryImageSource

import javax.imageio.ImageIO
import java.awt.Toolkit
import java.awt.image.BufferedImage
import br.gsj.utils.ImageUtils
import scala.collection.mutable.WrappedArray
import java.io.FileInputStream
import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.filter.EdgeFilter
import com.sksamuel.scrimage.nio.JpegWriter
import br.gsj.utils.MLUtils
import org.apache.spark.sql.DataFrame

object ImageSegmentation {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("imgSeg").master("local[*]").getOrCreate()
    val image_path = "src/main/resources/m1_blur.jpg"
    val result_path = image_path.substring(0, image_path.lastIndexOf("/") + 1) +
      image_path.substring(image_path.lastIndexOf("/") + 1, image_path.lastIndexOf(".")) +
      "_result" + image_path.substring(image_path.lastIndexOf("."), image_path.size)

    println(result_path)

    val final_image_df = ImageUtils.decodeImageDataFrame(spark, image_path)

    val features_col = Array(
      //            "w","h",
      "b", "g", "r")
    val vector_assembler = new VectorAssembler()
      .setInputCols(features_col)
      .setOutputCol("features")

    val va_transformed_df = vector_assembler.transform(final_image_df)

    val k = 15
    val kmeans = new KMeans().setK(k).setSeed(1L).setFeaturesCol("features")
    val kmeans_model = kmeans.fit(va_transformed_df)

    val va_cluster_df = kmeans_model.transform(va_transformed_df)
      .select("w", "h", "b", "g", "r", "prediction")

    var colors: Map[Int, (Int, Int, Int)] = Map[Int, (Int, Int, Int)]()

    for (i <- 0 to k - 1) {
      val s = va_cluster_df.filter(col("prediction") === i).rdd.first
      colors += (i -> (s(2).asInstanceOf[Int], s(3).asInstanceOf[Int], s(4).asInstanceOf[Int]))

    }

    val image_array_final = va_cluster_df.select("w", "h", "prediction")
      .rdd.map(f => (f.getAs[Int](0), f.getAs[Int](1), f.getAs[Int](2))).collect()

    val src_img = ImageIO.read(new File(image_path))
    val dest_img = ImageUtils.generateImage(src_img, image_array_final, colors)
    //
    ImageIO.write(dest_img, "jpg", new File(result_path))

  }

}