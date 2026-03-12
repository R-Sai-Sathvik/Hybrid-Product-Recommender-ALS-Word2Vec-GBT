# Hybrid Product Recommender System
### Apache Spark | ALS + Word2Vec + GBT | 20.9M Amazon Reviews

A distributed hybrid recommender system built on Apache Spark processing 20.9M Amazon 
reviews across 9.8M users — combining ALS collaborative filtering, Word2Vec content 
embeddings, and GBT re-ranking to serve personalized top-10 product recommendations.

## Dataset
- 20.9M ratings | 9.8M users | 756K items
- Amazon Electronics Product Reviews

## Pipeline
| Step | File | Tool | Output |
|------|------|------|--------|
| 1. Preprocess | Preprocess.scala | Spark SQL | ALS ratings, item index |
| 2. ALS Training | ALSTrainer.scala | SparkMLlib ALS | User/Item factor matrices |
| 3. Word2Vec | Word2VecEmbed2.scala | SparkMLlib W2V | 50-dim item embeddings |
| 4. GBT Ranker | TrainHybridModel.scala | SparkMLlib GBT | Hybrid ranker (AUC-ROC: 0.8245) |
| 5. Recommendations | GenerateRecommendations.scala | Spark | Top-10 per user |
| 6. Web UI | app.py + index.html | Python Flask | Dashboard + Search |

## Model Results
- AUC-ROC : 0.8245 
- Accuracy: 90.37% 
- Coverage: 100% of users

## Tech Stack
- Apache Spark 3.5.1 + Scala
- SparkMLlib (ALS, Word2Vec, GBT, VectorAssembler)
- HDFS (Hadoop Distributed File System)
- Docker (4-container simulated cluster)
- Python Flask + HTML/CSS/JS (Web UI)

## Cluster Setup
- namenode  : HDFS master (port 8020, 9870)
- datanode  : HDFS storage
- spark-master : Spark cluster manager (port 7077, 8080)
- spark-worker : 3 cores, 4GB RAM executor
