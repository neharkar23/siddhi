
from fastapi import APIRouter, HTTPException
from app.models import RAGRequest, RAGResponse
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm, get_llama_index_llm
from app.services.rag_chain import build_rag_retrieval_chain
from app.services.frameworks import get_agent
import logging
import asyncio

router = APIRouter()

@router.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    try:
        # Initialize component
        vector_store = get_vector_store(request.vector_store)

        if request.framework == "langgraph":
            llm = get_llm(request.llm_model)
            rag_chain = build_rag_retrieval_chain(llm, vector_store)
            agent = get_agent("langgraph", llm, rag_chain)

            inputs = {"messages": [("user", request.query)]}
            response_text = ""
            for step in agent.stream(inputs, stream_mode="values"):
                msg = step["messages"][-1]
                response_text += msg.content

            return RAGResponse(answer=response_text)

        elif request.framework == "llamaindex":
            llm = get_llama_index_llm(request.llm_model)
            rag_chain = build_rag_retrieval_chain(llm, vector_store)
            agent = get_agent("llamaindex", llm, rag_chain)

            async def main():
                return await agent.run(user_msg=request.query)

            response_text = asyncio.run(main())
            return RAGResponse(answer=str(response_text))

        elif request.framework == "dspy":
            llm = get_llm(request.llm_model)
            rag_chain = build_rag_retrieval_chain(llm, vector_store)
            agent = get_agent("dspy", llm, rag_chain)

            pred = agent(question=request.query)
            return RAGResponse(answer=str(pred.answer))

        else:
            raise HTTPException(status_code=400, detail="Invalid framework selected")

    except Exception as e:
        logging.exception("Error inside /ask:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


'''
from fastapi import APIRouter, HTTPException
from app.models import RAGRequest, RAGResponse
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm, get_llama_index_llm
from app.services.rag_chain import build_rag_retrieval_chain
from app.services.frameworks import get_agent
import logging
import asyncio

from prometheus_client import Counter, Summary  # ✅ Prometheus imports

router = APIRouter()

# ✅ Prometheus metrics
QUERY_COUNTER = Counter(
    "rag_queries_total", 
    "Total number of RAG queries received", 
    ["framework"]
)

REQUEST_LATENCY = Summary(
    "rag_request_latency_seconds", 
    "Time taken to process /ask request", 
    ["framework"]
)

@router.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    framework = request.framework

    QUERY_COUNTER.labels(framework).inc()              # ✅ Increment query counter
    with REQUEST_LATENCY.labels(framework).time():     # ✅ Measure latency
        try:
            vector_store = get_vector_store(request.vector_store)

            if framework == "langgraph":
                llm = get_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("langgraph", llm, rag_chain)

                inputs = {"messages": [("user", request.query)]}
                response_text = ""
                for step in agent.stream(inputs, stream_mode="values"):
                    msg = step["messages"][-1]
                    response_text += msg.content

                return RAGResponse(answer=response_text)

            elif framework == "llamaindex":
                llm = get_llama_index_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("llamaindex", llm, rag_chain)

                async def main():
                    return await agent.run(user_msg=request.query)

                response_text = asyncio.run(main())
                return RAGResponse(answer=str(response_text))

            elif framework == "dspy":
                llm = get_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("dspy", llm, rag_chain)

                pred = agent(question=request.query)
                return RAGResponse(answer=str(pred.answer))

            else:
                raise HTTPException(status_code=400, detail="Invalid framework selected")

        except Exception as e:
            logging.exception("Error inside /ask:")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
'''
'''
from fastapi import APIRouter, HTTPException
from app.models import RAGRequest, RAGResponse
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm, get_llama_index_llm
from app.services.rag_chain import build_rag_retrieval_chain
from app.services.frameworks import get_agent
import logging
import asyncio

from prometheus_client import Counter, Summary  # ✅ Prometheus imports
from opentelemetry import logs
from opentelemetry._logs import LoggerProvider, SeverityNumber
from opentelemetry.sdk._logs import LoggingHandler

router = APIRouter()

# ✅ Prometheus metrics
QUERY_COUNTER = Counter(
    "rag_queries_total", 
    "Total number of RAG queries received", 
    ["framework"]
)

REQUEST_LATENCY = Summary(
    "rag_request_latency_seconds", 
    "Time taken to process /ask request", 
    ["framework"]
)

# ✅ OTEL logger setup
logger = logging.getLogger("otel_logger")
logger.setLevel(logging.INFO)
handler = LoggingHandler(level=logging.INFO)
logger.addHandler(handler)

@router.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    framework = request.framework

    QUERY_COUNTER.labels(framework).inc()              # ✅ Increment query counter
    with REQUEST_LATENCY.labels(framework).time():     # ✅ Measure latency
        try:
            vector_store = get_vector_store(request.vector_store)

            if framework == "langgraph":
                llm = get_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("langgraph", llm, rag_chain)

                inputs = {"messages": [("user", request.query)]}
                response_text = ""
                for step in agent.stream(inputs, stream_mode="values"):
                    msg = step["messages"][-1]
                    response_text += msg.content

                logger.info(f"Input: {request.query} | Framework: {framework} | Vector Store: {request.vector_store}")
                logger.info(f"Output: {response_text}")

                return RAGResponse(answer=response_text)

            elif framework == "llamaindex":
                llm = get_llama_index_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("llamaindex", llm, rag_chain)

                async def main():
                    return await agent.run(user_msg=request.query)

                response_text = asyncio.run(main())
                logger.info(f"Input: {request.query} | Framework: {framework} | Vector Store: {request.vector_store}")
                logger.info(f"Output: {response_text}")

                return RAGResponse(answer=str(response_text))

            elif framework == "dspy":
                llm = get_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("dspy", llm, rag_chain)

                pred = agent(question=request.query)
                logger.info(f"Input: {request.query} | Framework: {framework} | Vector Store: {request.vector_store}")
                logger.info(f"Output: {pred.answer}")

                return RAGResponse(answer=str(pred.answer))

            else:
                raise HTTPException(status_code=400, detail="Invalid framework selected")

        except Exception as e:
            logging.exception("Error inside /ask:")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
'''
'''

from fastapi import APIRouter, HTTPException
from app.models import RAGRequest, RAGResponse
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm, get_llama_index_llm
from app.services.rag_chain import build_rag_retrieval_chain
from app.services.frameworks import get_agent
import logging
import asyncio

# Prometheus Metrics
from prometheus_client import Counter, Summary

# OpenTelemetry Logging
from opentelemetry.sdk.logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk.logs.export import BatchLogProcessor, OTLPLogExporter
from opentelemetry import logs

router = APIRouter()

# ---------- Prometheus Setup ----------
QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total number of RAG queries received",
    ["framework"]
)

REQUEST_LATENCY = Summary(
    "rag_request_latency_seconds",
    "Time taken to process /ask request",
    ["framework"]
)

# ---------- OpenTelemetry Logging Setup ----------
exporter = OTLPLogExporter(endpoint="http://localhost:4317", insecure=True)

log_provider = LoggerProvider()
log_processor = BatchLogProcessor(exporter)
log_provider.add_log_processor(log_processor)
logs.set_logger_provider(log_provider)

otel_logger = logging.getLogger("otel_logger")
otel_logger.setLevel(logging.INFO)
otel_logger.addHandler(LoggingHandler(level=logging.INFO, logger_provider=log_provider))

# ---------- FastAPI Route ----------
@router.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    framework = request.framework

    QUERY_COUNTER.labels(framework).inc()
    with REQUEST_LATENCY.labels(framework).time():
        try:
            vector_store = get_vector_store(request.vector_store)

            if framework == "langgraph":
                llm = get_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("langgraph", llm, rag_chain)

                inputs = {"messages": [("user", request.query)]}
                response_text = ""
                for step in agent.stream(inputs, stream_mode="values"):
                    msg = step["messages"][-1]
                    response_text += msg.content

            elif framework == "llamaindex":
                llm = get_llama_index_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("llamaindex", llm, rag_chain)

                async def main():
                    return await agent.run(user_msg=request.query)

                response_text = asyncio.run(main())

            elif framework == "dspy":
                llm = get_llm(request.llm_model)
                rag_chain = build_rag_retrieval_chain(llm, vector_store)
                agent = get_agent("dspy", llm, rag_chain)

                pred = agent(question=request.query)
                response_text = pred.answer

            else:
                raise HTTPException(status_code=400, detail="Invalid framework selected")

            # ✅ OTEL log input and output
            otel_logger.info(f"[Input] Query: {request.query} | Framework: {framework} | VectorStore: {request.vector_store}")
            otel_logger.info(f"[Output] Answer: {response_text}")

            return RAGResponse(answer=str(response_text))

        except Exception as e:
            otel_logger.error(f"[Error] Exception in /ask: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
'''