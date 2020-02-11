package br.gsj.utils

import javax.imageio.ImageIO

import java.awt.image.BufferedImage
import java.io.File
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import java.awt.image.WritableRaster
import java.awt.Color
import scala.collection.mutable.WrappedArray

case class PixelInfo(pixel: Int, a: Int, r: Int, g: Int, b: Int, x: Int, y: Int)
case class Pixel(a: Int, r: Int, g: Int, b: Int, x: Int, y: Int)

object ImageUtils {

  def loadImageArray(path: String): Array[PixelInfo] = {
    ImageIO.setUseCache(false)
    val image = ImageIO.read(new File(path))

    // obtain width and height of image
    val w = image.getWidth
    val h = image.getHeight

    var array = Array[PixelInfo]()
    var cont = 1
    for (x <- 0 until w)
      for (y <- 0 until h) {
        val argba = printPixelARGB(image.getRGB(x, y))
        //println(x+","+y)
        array = array :+ PixelInfo(image.getRGB(x, y), argba._1, argba._2, argba._3, argba._4, x, y)
        cont += 1
      }
    array

  }

  /**
   *
   */

  def decodeImageDataFrame(spark: SparkSession, path: String): DataFrame = {

    val image_df = spark.read.format("image").load(path)

    val rdd_img_array = image_df.select(col("image.data"))
      .rdd.flatMap(f => f.getAs[Array[Byte]]("data"))

    val img_data = image_df.select(col("image.*"))
      .rdd.map(row => (
        row.getAs[Int]("height"),
        row.getAs[Int]("width"),
        row.getAs[Int]("nChannels")))
      .collect()(0)

    val height = img_data._1
    val width = img_data._2
    val nChannels = img_data._3

    var offSet = spark.sparkContext.longAccumulator("offSetAcc")
    var x = spark.sparkContext.longAccumulator("xAcc")
    var y = spark.sparkContext.longAccumulator("yAcc")
    x.add(1)
    y.add(1)

    import spark.implicits._
    var final_image_df = rdd_img_array.zipWithIndex().map { f =>

      if (offSet.value == 0) {
        //b
        offSet.add(1)
        if (f._2 != 0)
          x.add(1)
      } else if (offSet.value == 1) {
        //g
        offSet.add(1)
      } else if (offSet.value == 2) {
        //r
        offSet.reset()
      }
      if (x.value == (width)) {
        x.reset()
        y.add(1)
      }

      (f._1 & 0xFF, x.value, y.value)
    }.toDF.withColumnRenamed("_1", "color")
      .withColumnRenamed("_2", "w").withColumnRenamed("_3", "h")

    import spark.implicits._

    final_image_df.groupBy(col("w"), col("h"))
      .agg(collect_list(col("color")).as("color"))
      .orderBy(col("w"), col("h"))
      .rdd
      .map { f =>
        val a = f(2).asInstanceOf[WrappedArray[Int]]
        (f(0).toString.toInt, f(1).toString.toInt, a(0), a(1), a(2))
      }
      .toDF
      .withColumnRenamed("_1", "w")
      .withColumnRenamed("_2", "h")
      .withColumnRenamed("_3", "b")
      .withColumnRenamed("_4", "g")
      .withColumnRenamed("_5", "r")

  }

  def printPixelARGB(pixel: Int): (Int, Int, Int, Int) = {
    val alpha = (pixel >> 24) & 0xff;
    val red = (pixel >> 16) & 0xff;
    val green = (pixel >> 8) & 0xff;
    val blue = (pixel) & 0xff;
    (alpha, red, green, blue)
  }

  def generateImage(img: BufferedImage, img_df: DataFrame): BufferedImage = {

    val width = img.getWidth
    val height = img.getHeight

    println("Output width: " + width)
    println("Output height: " + height)
    
    img_df.show

    val array_img = img_df.rdd.map(f => (f(0).asInstanceOf[Int], f(1).asInstanceOf[Int],
      f(2).asInstanceOf[Int], f(3).asInstanceOf[Int],
      f(4).asInstanceOf[Int])).collect

    val out = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)

    for (p <- array_img) {
      out.setRGB(p._1, p._2, new Color(p._5,p._4,p._3).getRGB)
    }

    out

  }

  def generateImage(img: BufferedImage, image_array: Array[(Int, Int, Int)], colors: Map[Int, (Int, Int, Int)]): BufferedImage = {
    // obtain width and height of image
    val width = img.getWidth
    val height = img.getHeight

    println("Output width: " + width)
    println("Output height: " + height)

    if (width * height != image_array.size)
      throw new IllegalArgumentException("image array does not fit the provided image");

    // create new image of the same size
    val out = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)

    for (j <- 0 until image_array.size - 1) {
      //      println(j + ": " + image_array(j)._1 + " " + image_array(j)._2)
      val color_tuple = colors(image_array(j)._3)
      out.setRGB(image_array(j)._1, image_array(j)._2,
        new Color(color_tuple._3, color_tuple._2, color_tuple._1).getRGB)
    }
    out
  }

  def phototest(img: BufferedImage): BufferedImage = {
    // obtain width and height of image
    val w = img.getWidth
    val h = img.getHeight

    // create new image of the same size
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)

    // copy pixels (mirror horizontally)
    for (x <- 0 until w)
      for (y <- 0 until h)
        out.setRGB(x, y, img.getRGB(x, y))

    // draw red diagonal line
    //    for (x <- 0 until (h min w))
    //      out.setRGB(x, x, 0xff0000)

    out
  }


}