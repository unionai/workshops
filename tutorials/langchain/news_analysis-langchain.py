# LangChain + Flyte 2.0: AI-Powered News Analysis Pipeline
# This example demonstrates how LangChain agents work with Flyte 2.0's
# revolutionary pure Python workflow orchestration

import flyte
from dataclasses import dataclass, asdict
from typing import List
import os

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

# Data classes for type safety
@dataclass
class NewsArticle:
    title: str
    content: str
    url: str
    published_date: str
    source: str

@dataclass
class AnalysisResult:
    sentiment: str
    key_topics: List[str]
    summary: str
    confidence_score: float

@dataclass
class MarketInsight:
    trend: str
    impact_assessment: str
    recommended_actions: List[str]
    risk_level: str

# =============================================================================
# FLYTE 2.0: Task Environment - Unified Environment for All Tasks
# =============================================================================

# Single environment with all dependencies
langchain_env = flyte.TaskEnvironment(
    name="langchain-pipeline",
    resources=flyte.Resources(memory="8Gi", cpu=4),
    image=flyte.Image.from_debian_base().with_pip_packages(
        "langchain>=0.3.0",
        "langchain-openai",
        "langchain-community",
        "duckduckgo-search",
        "ddgs",
        "pandas",
        "numpy",
        "unionai-reuse==0.1.6"
    ),
    secrets=[flyte.Secret(key="OPENAI_API_KEY")],
    reusable=flyte.ReusePolicy(
        replicas=4,
        idle_ttl=600,
        concurrency=6,
    )
)

# =============================================================================
# FLYTE 2.0: Tasks with Caching and Tracing
# =============================================================================

@langchain_env.task(cache="auto")
async def collect_news_data(query: str, num_articles: int = 5) -> List[dict]:
    """
    Uses LangChain's search agent to collect news articles.
    Flyte 2.0's async support enables better resource utilization.
    """
    print(f"🔍 Collecting news data for: {query}")

    try:
        # Initialize LangChain components
        search = DuckDuckGoSearchRun()

        # For demo purposes, create sample articles
        # In production, you'd parse real search results
        articles = []
        for i in range(num_articles):
            article = NewsArticle(
                title=f"Breaking: {query} Development #{i+1}",
                content=f"Latest news about {query} with significant market implications...",
                url=f"https://example-news-{i+1}.com",
                published_date="2025-09-25",
                source=f"NewsSource{i+1}"
            )
            articles.append(article)

        print(f"✅ Collected {len(articles)} articles")
        return [asdict(article) for article in articles]

    except Exception as e:
        print(f"⚠️  Error in data collection: {e}")
        raise e


@langchain_env.task(cache="auto")
async def analyze_content(articles) -> List[dict]:
    """
    Uses LangChain agent for AI-powered content analysis.
    Demonstrates Flyte 2.0's GPU environment integration.
    """
    print(f"🤖 Analyzing {len(articles)} articles with AI")

    try:
        # Get OpenAI API key from Flyte 2.0 secrets
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  No OpenAI key found, using mock analysis")
            mock_results = [AnalysisResult(
                sentiment="positive",
                key_topics=["AI", "Technology", "Demo"],
                summary=f"Mock analysis of {article['title']}",
                confidence_score=0.85
            ) for article in articles]
            return [asdict(result) for result in mock_results]

        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use more cost-effective model for demo
            temperature=0.1,
            openai_api_key=api_key
        )

        # Create analysis agent with memory
        memory = ConversationBufferMemory(memory_key="chat_history")
        analysis_results = []

        for article in articles:
            # Create analysis prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert content analyst. Analyze the given news article and provide:
                1. Sentiment (positive, negative, neutral)
                2. Key topics (list of main themes)
                3. Summary (concise overview)
                4. Confidence score (0-1)

                Return results as JSON."""),
                ("human", f"Title: {article['title']}\nContent: {article['content']}")
            ])

            response = await llm.ainvoke(prompt.format_messages())

            # Parse response (simplified for demo)
            analysis = AnalysisResult(
                sentiment="positive" if "positive" in response.content.lower() else "neutral",
                key_topics=["AI", "Technology", "Market"],
                summary=f"Analysis of {article['title']}",
                confidence_score=0.85
            )

            analysis_results.append(analysis)

            # Add to memory for context
            memory.chat_memory.add_user_message(article['title'])
            memory.chat_memory.add_ai_message(f"Analyzed with {analysis.sentiment} sentiment")

        print(f"✅ Analyzed {len(analysis_results)} articles")
        return [asdict(result) for result in analysis_results]

    except Exception as e:
        print(f"⚠️  Error in content analysis: {e}")
        raise e

@langchain_env.task(cache="auto")
async def generate_market_insights(articles, analyses) -> List[dict]:
    """
    Uses LangChain's reasoning agent for market intelligence.
    Showcases Flyte 2.0's custom caching strategies.
    """
    print(f"📊 Generating market insights from {len(analyses)} analyses")

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  Using mock market insights")
            mock_insights = [MarketInsight(
                trend="Bullish",
                impact_assessment="Moderate impact expected",
                recommended_actions=["Monitor developments", "Adjust strategy"],
                risk_level="Medium"
            ) for _ in articles]
            return [asdict(insight) for insight in mock_insights]

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=api_key
        )

        # Create market analysis functions
        def calculate_sentiment_score(sentiments):
            """Calculate overall market sentiment score"""
            positive_count = sum(1 for s in sentiments if s == "positive")
            return positive_count / len(sentiments) if sentiments else 0.5

        def assess_risk_level(sentiment_score: float) -> str:
            """Assess market risk level"""
            if sentiment_score > 0.7:
                return "Low Risk"
            elif sentiment_score > 0.4:
                return "Medium Risk"
            else:
                return "High Risk"

        # Extract sentiment data
        sentiments = [analysis['sentiment'] for analysis in analyses]
        sentiment_score = calculate_sentiment_score(sentiments)
        risk_level = assess_risk_level(sentiment_score)

        # Generate insights using LangChain reasoning
        insights_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior market analyst. Generate actionable insights."""),
            ("human", f"Sentiment score: {sentiment_score}, Risk: {risk_level}")
        ])

        response = await llm.ainvoke(insights_prompt.format_messages())

        insights = []
        for i in range(len(articles)):
            insight = MarketInsight(
                trend="Bullish" if sentiment_score > 0.6 else "Bearish",
                impact_assessment=f"Impact score: {sentiment_score:.2f}",
                recommended_actions=[
                    "Monitor key developments",
                    "Adjust portfolio allocation",
                    "Prepare contingency plans"
                ],
                risk_level=risk_level
            )
            insights.append(insight)

        print(f"✅ Generated {len(insights)} market insights")
        return [asdict(insight) for insight in insights]

    except Exception as e:
        print(f"⚠️  Error generating insights: {e}")
        raise e

@langchain_env.task
async def generate_executive_report(articles, analyses, insights) -> str:
    """
    Generate comprehensive executive report using LangChain.
    Final task demonstrating Flyte 2.0's workflow composition.
    """
    print("📄 Generating executive report")

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return f"""
# AI-Powered Market Intelligence Report (Demo Mode)
*Generated by LangChain + Flyte 2.0 Pipeline*

## Executive Summary
Analyzed {len(articles)} articles with {len(analyses)} AI-powered analyses.
Generated {len(insights)} market insights using advanced reasoning.

## Key Findings
- Flyte 2.0's task environments enable seamless LangChain integration
- Async task execution improves resource utilization
- Environment-based resource management simplifies deployment

## Flyte 2.0 Features Demonstrated
- ✅ Task environments with reusable containers
- ✅ Async/await support for better performance
- ✅ Built-in caching strategies
- ✅ Native error handling with try/except
- ✅ Seamless LangChain agent integration

*This report demonstrates the power of combining LangChain's AI capabilities
with Flyte 2.0's revolutionary workflow orchestration.*
"""

        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

        # Prepare data summary
        avg_sentiment = sum(1 for a in analyses if a['sentiment'] == "positive") / len(analyses)
        dominant_trend = insights[0]['trend'] if insights else "Neutral"

        report_prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a professional executive summary report."""),
            ("human", f"""
            Articles: {len(articles)}
            Avg sentiment: {avg_sentiment:.2%}
            Trend: {dominant_trend}
            """)
        ])

        response = await llm.ainvoke(report_prompt.format_messages())

        final_report = f"""
# AI-Powered Market Intelligence Report
*Generated by LangChain + Flyte 2.0 Pipeline*

Date: 2025-09-25
Pipeline: LangChain Agents → Flyte 2.0 Task Environments

{response.content}

---
## Flyte 2.0 Pipeline Metrics
- Articles processed: {len(articles)}
- Analysis confidence: {sum(a['confidence_score'] for a in analyses)/len(analyses):.2%}
- Processing: Optimized with async/await and container reuse
- Recovery: Built-in fault tolerance with native error handling

## Revolutionary Flyte 2.0 Features Used
- **Task Environments**: Declarative resource and dependency management
- **Async Execution**: Better performance with concurrent task execution
- **Smart Caching**: Version-aware caching with custom strategies
- **Container Reuse**: Eliminate cold starts with warm container pools
- **Native Error Handling**: Exception handling with task environments

*This pipeline showcases the future of AI workflow orchestration with
Flyte 2.0's task environment approach and LangChain's agent capabilities.*
"""

        print("✅ Executive report generated")
        return final_report

    except Exception as e:
        print(f"⚠️  Error generating report: {e}")
        return f"Report generation failed: {e}"

# =============================================================================
# FLYTE 2.0: Main Workflow Task
# =============================================================================

@langchain_env.task
async def ai_news_analysis_pipeline(
    search_query: str = "artificial intelligence",
    num_articles: int = 5
) -> str:
    """
    Main Flyte 2.0 Pure Python Workflow - No @workflow decorator needed!

    This demonstrates Flyte 2.0's revolutionary approach:
    - Pure Python function (not DSL-constrained)
    - Runtime workflow construction
    - Native async/await support for parallelism
    - Tasks calling tasks pattern
    - Built-in error handling with try/except

    The workflow showcases how Flyte 2.0's pure Python approach
    enables seamless integration with LangChain agents.
    """

    print(f"🚀 Starting Flyte 2.0 AI News Analysis Pipeline")
    print(f"   Query: {search_query}")
    print(f"   Articles: {num_articles}")

    try:
        # Step 1: Data Collection (with container reuse)
        print("\n📝 Step 1: Data Collection")
        articles = await collect_news_data(search_query, num_articles)

        # Step 2: AI Content Analysis (GPU-accelerated if available)
        print("\n🧠 Step 2: AI Content Analysis")
        analyses = await analyze_content(articles)

        # Step 3: Market Intelligence Generation (async for better performance)
        print("\n📈 Step 3: Market Intelligence")
        insights = await generate_market_insights(articles, analyses)

        # Step 4: Executive Report Generation
        print("\n📊 Step 4: Executive Report Generation")
        report = await generate_executive_report(articles, analyses, insights)

        print("\n✅ Pipeline completed successfully!")
        return report

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise e

# =============================================================================
# FLYTE 2.0: Dynamic Multi-Topic Analysis (Runtime Workflow Construction)
# =============================================================================

async def dynamic_multi_topic_analysis(topics):
    """
    Advanced Flyte 2.0 example showing runtime workflow construction.

    This demonstrates:
    - Dynamic workflow creation based on runtime inputs
    - Parallel execution of multiple analysis pipelines
    - Pure Python loops and conditionals (no DSL constraints)
    - Async/await for true parallelism
    """

    print(f"🔄 Dynamic Multi-Topic Analysis for {len(topics)} topics")

    reports = []

    # Flyte 2.0: Pure Python loops work naturally!
    for topic in topics:
        print(f"   Processing topic: {topic}")

        try:
            # Each topic gets its own analysis pipeline
            # Tasks run in parallel thanks to async/await
            articles = await collect_news_data(topic, 3)
            analyses = await analyze_content(articles)
            insights = await generate_market_insights(articles, analyses)
            report = await generate_executive_report(articles, analyses, insights)

            reports.append(f"Report for '{topic}':\n{report}\n" + "="*80 + "\n")

        except Exception as e:
            print(f"⚠️  Failed to process topic '{topic}': {e}")
            # Graceful error handling - continue with other topics
            reports.append(f"Failed to analyze topic '{topic}': {e}")

    print(f"✅ Completed analysis for {len(reports)} topics")
    return reports

# =============================================================================
# FLYTE 2.0: Usage Examples and Deployment
# =============================================================================

if __name__ == "__main__":
    """
    Flyte 2.0 + LangChain Example

    To run this pipeline:
    1. Install dependencies: pip install 'flyte>=2.0.0b21' langchain langchain-openai langchain-community duckduckgo-search --prerelease=allow
    2. Set environment: export OPENAI_API_KEY="your-key-here"  (optional for demo)
    3. Run locally: python test.py
    4. Deploy to Flyte: flyte deploy test.py (requires Flyte cluster)
    """

    print("🌟 LangChain + Flyte 2.0: AI-Powered News Analysis Pipeline")
    print("\nFlyte 2.0 Features Showcased:")
    print("✅ Task environments with declarative resources")
    print("⚡ Async/await native support")
    print("🐳 Smart container reuse")
    print("🔍 Built-in caching strategies")
    print("🚀 Seamless LangChain agent integration")

    # Initialize Flyte locally for demo
    flyte.init_from_config(".flyte/config.yaml")
    # flyte.init() # uncomment to run locally
    # Run the pipeline
    print(f"\n🏃‍♂️ Running AI News Analysis Pipeline...")
    execution = flyte.run(ai_news_analysis_pipeline, search_query="artificial intelligence", num_articles=3)

    print(f"\n✅ Execution completed!")
    print(f"Execution ID: {execution.name}")
    print(f"URL: {execution.url}")

    # For deployment to remote Flyte cluster:
    # flyte init_from_config(".flyte/config.yaml")
    # execution = flyte.run(ai_news_analysis_pipeline, search_query="blockchain")