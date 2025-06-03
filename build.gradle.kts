plugins {
    application
    kotlin("jvm") version "2.1.20"
    // Scala
    id("org.gradle.scala")
    // 打包 fat-jar
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "xyz.ifilk"
version = "1.0-SNAPSHOT"


val scalaMajor = "2.13"
val sparkVersion = "3.5.6"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.apache.spark:spark-core_2.13:3.5.6")
    implementation("org.apache.spark:spark-sql_2.12:3.5.6")
    implementation("org.apache.hbase:hbase-mapreduce:2.6.2")
    implementation("org.apache.hbase:hbase-client:2.6.2")
    implementation("org.apache.hbase:hbase-common:2.6.2")
    // Scala 库（Spark 已传递，但在 IDE 里显式写出更直观）
    implementation("org.scala-lang:scala-library:2.13.13")

    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    // Spark >3.3 支持 Java 17）
    jvmToolchain(17)
}
/* ---------- Scala ---------- */
tasks.withType<ScalaCompile>().configureEach {
    classpath += files("build/classes/kotlin/main")
    scalaCompileOptions.apply {
        targetCompatibility = "17"
        sourceCompatibility = "17"
        additionalParameters = listOf("-deprecation", "-feature")
    }
}
// 先编译 Kotlin 后编译Scala
tasks.compileScala {
    dependsOn(tasks.compileKotlin)
}

/* ---------- Shadow-Jar ---------- */
tasks.shadowJar {
    archiveFileName.set("app.jar")
    archiveBaseName.set("app")
    archiveClassifier.set("")   // 生成 <name>-<version>.jar 而非 *-all.jar
    isZip64 = true
}

/* ---------- 生成可执行入口 ---------- */
application {
    mainClass.set("xyz.ifilk.dataset.mnist.MNISTTrain")
}
