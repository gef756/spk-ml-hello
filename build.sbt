name := "spk-ml-hello"

version := "1.0"

scalaVersion := "2.10.4"

assemblyMergeStrategy in assembly <<= (assemblyMergeStrategy in assembly) {
  old => {
    case PathList("META-INF", xs @_*) => MergeStrategy.discard
    case x => MergeStrategy.first
  }
}

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.4.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.4.1" % "provided"