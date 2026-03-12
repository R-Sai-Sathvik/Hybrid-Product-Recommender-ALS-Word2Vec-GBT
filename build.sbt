name := "bda-project"
version := "1.0"
scalaVersion := "2.12.18"

val sparkVersion = "3.5.1"

Compile / scalaSource := baseDirectory.value / "scala" / "src" / "main" / "scala"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"   % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
)

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", _*) => MergeStrategy.discard
  case _                        => MergeStrategy.first
}