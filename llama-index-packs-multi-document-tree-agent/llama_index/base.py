"""Multi-document agents Pack."""

from typing import Any, Dict, List

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import ServiceContext, SummaryIndex, VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.schema import Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import (
    ObjectIndex,
    SimpleToolNodeMapping,
    ObjectRetriever,
)
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai import OpenAI
import json

class HumanInputRequiredException(Exception):
    """Exception raised when human input is required."""

    def __init__(
        self,
        message="Human input is required",
    ):
        self.message = message
        super().__init__(self.message)


def humanInputRequired(input: str) -> int:
    """Raises a Human Input Required Exception"""
    raise HumanInputRequiredException(message=input)


# define a custom object retriever that adds in a query planning tool
class DocumentStructureCustomRetriever(ObjectRetriever):
    def __init__(self, retriever, object_node_mapping, all_tools, llm=None):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OpenAI("gpt-4")

    def retrieve(self, query_bundle):
        nodes = self._retriever.retrieve(query_bundle)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, llm=self._llm
        )
        sub_question_description = f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]

ab = [[1, 2], [2, 3]]
ab = [item for sublist in ab for item in sublist]

        
class MultiDocumentTreeAgentsPack(BaseLlamaPack):
    """Multi-document Tree Agents pack.

    Given a set of documents, build our multi-document agents architecture.
    - setup a document agent over agent doc
      - capable of document level vector topk search and summarization
    - setup a top-level agent over doc agents:
      - it can use ObjectIndex to fetch the most relevant documents (global-vector-topk-search)
      - it can scroll the TOC and link click into a particular document
    - Questions:
      - Best way to encode the Table Of Content?
        - Pass relevant TOC section/chunk into the doc as metadata
        - have it be a tool that's available by the top level agent to reference
        - should we have another LLM to decide whether or not to scroll the TOC to find relevant things
    """

    def __init__(
        self,
        docs: List[Document],
        doc_titles: List[str],
        doc_descriptions: List[str],
        metadata: Document,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.node_parser = SentenceSplitter()
        Settings.llm = OpenAI(temperature=0, model="gpt-4")

        humanInputTool = FunctionTool.from_defaults(
            fn=humanInputRequired,
            tool_metadata=ToolMetadata(
                name="UserInputTool",
                description="Use this tool if user's input is required to answer the query",
            ),
        )

        # Embed title and link structure into Node
        kwargs["metadata"] = kwargs.get("metadata", {})
        with open("structure.json", "r") as f:
            structure = json.load(f)

        # Build agents dictionary
        self.agents = {}

        # this is for the baseline
        all_nodes = []

        # build agent for each document
        for idx, doc in enumerate(docs):
            doc_title = doc_titles[idx]
            doc_description = doc_descriptions[idx]
            nodes = self.node_parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)

            # build vector index
            vector_index = VectorStoreIndex(nodes, service_context=self.service_context)
            summary_index = SummaryIndex(nodes, service_context=self.service_context)

            # define query engines
            vector_query_engine = vector_index.as_query_engine()
            summary_query_engine = summary_index.as_query_engine()

            # define tools
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=vector_query_engine,
                    metadata=ToolMetadata(
                        name="vector_tool",
                        description=(
                            "Useful for questions related to specific aspects of"
                            f" {doc_title}."
                        ),
                    ),
                ),
                QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING about {doc_title}. "
                        ),
                    ),
                ),
                humanInputTool,                
            ]

            # build agent
            agent = ReActAgent.from_tools(
                query_engine_tools,
                verbose=True,
                llm=OpenAI(model="gpt-4"),
            )

            self.agents[doc_title] = agent

        # build top-level, retrieval-enabled OpenAI Agent
        # define tool for each document agent
        all_tools = []
        for idx, doc in enumerate(docs):
            doc_title = doc_titles[idx]
            doc_description = doc_descriptions[idx]
            wiki_summary = (
                f"Use this tool if you want to answer any questions about {doc_title}.\n"
                f"Doc description: {doc_description}\n"
            )
            doc_tool = QueryEngineTool(
                query_engine=self.agents[doc_title],
                metadata=ToolMetadata(
                    name=f"tool_{doc_title}",
                    description=wiki_summary,
                ),
            )
            all_tools.append(doc_tool)

        tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
        self.obj_index = ObjectIndex.from_objects(
            all_tools,
            tool_mapping,
            VectorStoreIndex,
        )

        # topk retrieval
        self.top_agent = OpenAIAgent.from_tools(
            tool_retriever=self.obj_index.as_retriever(similarity_top_k=3),
            system_prompt=""" \
You are an agent designed to answer queries about a set of given documents.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
        """,
            verbose=True,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "top_agent": self.top_agent,
            "obj_index": self.obj_index,
            "doc_agents": self.agents,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.top_agent.query(*args, **kwargs)


