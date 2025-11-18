from diagrams import Diagram, Cluster
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.analytics import Spark
from diagrams.onprem.client import Users
from diagrams.onprem.monitoring import Grafana
from diagrams.programming.language import Python
from diagrams.custom import Custom

with Diagram("RAG Case Study Architecture", show=True, direction="TB"):

    user = Users("User Query")

    with Cluster("Document Processing"):
        pdf = Custom("PDF Documents", "./icons/pdf.png")
        chunking = Python("Chunking")
        embedding = Python("Embeddings (all-mpnet-base-v2)")
        vector_store = PostgreSQL("Vector Store")

        pdf >> chunking >> embedding >> vector_store

    with Cluster("Retrieval"):
        retrieval = Spark("Top-k Retrieval")

    with Cluster("Generation"):
        claude = Server("Claude")
        gemini = Server("Gemini")
        openai = Server("OpenAI GPT")

    with Cluster("Evaluation"):
        auto_qs = Python("Auto Qs (249)")
        golden = Python("Golden Dataset (15)")
        metrics = Grafana("Metrics")

        auto_qs >> metrics
        golden >> metrics

    with Cluster("Dashboard"):
        dashboard = Custom("Streamlit UI", "./icons/streamlit.png")

    # Flow
    user >> retrieval
    vector_store >> retrieval >> [claude, gemini, openai] >> dashboard
    metrics >> dashboard
