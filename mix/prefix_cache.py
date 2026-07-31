#########
### Check anthropic (via OpenRouter) and DeepSeek cached tokens
#########

# Requires: openai>=1.30.0
import os
from anthropic import Anthropic

api_key = os.environ["ANTHROPIC_API_KEY"]
client = Anthropic(api_key=api_key)

# Large system prompt with a document corpus — cacheable prefix.
# Needs to clear ~1024 tokens for caching to actually engage on either
# provider; the original placeholder text was ~30 tokens, far too short
# to ever produce a cache hit.
DOC_SECTION = """Section {n}: Rate Limits and Authentication
This service enforces a rolling rate limit of 500 requests per minute per API key.
Exceeding the limit returns HTTP 429 with a Retry-After header indicating the
number of seconds to wait before retrying. Authentication failures return HTTP 401
and do not count against the rate limit. All requests must include a valid
Authorization header formatted as "Bearer <token>". Tokens expire after 24 hours
and must be refreshed via the /auth/refresh endpoint before expiry.
"""
SYSTEM_PROMPT = (
    "You are a technical documentation assistant. Answer questions "
    "based only on the provided documentation. Be precise and cite section numbers.\n\n"
    + "\n".join(DOC_SECTION.format(n=i) for i in range(1, 61))
)


def print_cache_usage(usage) -> None:
    print(f"  prompt_tokens: {getattr(usage, 'prompt_tokens', None)}")
    print(f"  completion_tokens: {getattr(usage, 'completion_tokens', None)}")
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        print(f"  cached_tokens: {getattr(details, 'cached_tokens', None)}")
    print(f"  raw usage: {usage}")


def query_with_prompt_cache(user_question: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Mark as cacheable
            }
        ],
        messages=[
            {"role": "user", "content": user_question},
        ],
    )
    print_cache_usage(response.usage)
    return response.content[0].text


# First call: populates the cache (pays cache creation cost)
print("cc_answer1:\n")
answer1 = query_with_prompt_cache("What are the rate limits?")
print("\n")
# Subsequent calls with the same system prompt: should hit cache
print("cc_answer2:\n")
answer2 = query_with_prompt_cache("How do I handle authentication errors?")
print("\n")
print("cc_answer3:\n")
answer3 = query_with_prompt_cache("What is the maximum token limit per request?")
print("\n")

#########
### Check ds cached token
#########
from openai import OpenAI

api_key = os.environ["DEEPSEEK_API_KEY"]
client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)

def query_with_openai_cache(user_question: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},  # Cached prefix
            {"role": "user", "content": user_question},
        ],
    )
    print_cache_usage(response.usage)
    return response.choices[0].message.content

print("ds_answer1:\n")
ds_answer1 = query_with_openai_cache("What are the rate limits?")
print("\n")
print("ds_answer2:\n")
ds_answer2 = query_with_openai_cache("How do I handle authentication errors?")
print("\n")

if False:
    #########
    ### Redis Stack (available as redis/redis-stack on Docker Hub) adds vector similarity search to standard Redis
    #########
    # Requires: redis>=5.0.0 (with Redis Stack / redis-stack-server), numpy>=1.26.0
    from redis import Redis
    from redis.commands.search.field import VectorField, TextField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    import numpy as np
    import json

    r = Redis(host="localhost", port=6379)

    # Create vector index on first run
    VECTOR_DIM = 1536  # text-embedding-3-small dimension

    try:
        r.ft("cache_index").create_index(
            [
                TextField("query_text"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
            ],
            definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
        )
        print("Vector index created.")
    except Exception:
        pass  # Index already exists

    def vector_cache_lookup(query_embedding: list[float], threshold: float = 0.93) -> str | None:
        embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        query = (
            Query("*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("query_text", "response", "score")
            .dialect(2)
        )

        results = r.ft("cache_index").search(
            query, query_params={"vec": embedding_bytes}
        )

        if results.total > 0:
            top = results.docs[0]
            similarity = 1.0 - float(top.score)  # COSINE distance → similarity
            if similarity >= threshold:
                return top.response

        return None

    #########
    ### Monitoring and hit rate tracking
    #########

    # Requires: Python>=3.10 (stdlib only)
    import time
    from dataclasses import dataclass, field
    from collections import defaultdict

    @dataclass
    class CacheMetrics:
        hits: int = 0
        misses: int = 0
        total_latency_ms: float = 0.0
        hit_latency_ms: float = 0.0
        miss_latency_ms: float = 0.0

        @property
        def hit_rate(self) -> float:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

        @property
        def avg_latency_ms(self) -> float:
            total = self.hits + self.misses
            return self.total_latency_ms / total if total > 0 else 0.0

    metrics = CacheMetrics()

    def tracked_cache_call(query: str) -> tuple[str, bool]:
        start = time.perf_counter()
        cached = semantic_cache_lookup(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if cached:
            metrics.hits += 1
            metrics.hit_latency_ms += elapsed_ms
            metrics.total_latency_ms += elapsed_ms
            return cached, True

        # Cache miss — full LLM call
        response = cached_llm_call(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.misses += 1
        metrics.miss_latency_ms += elapsed_ms
        metrics.total_latency_ms += elapsed_ms
        return response, False

    # Log metrics every N requests or to your observability platform
    def log_metrics():
        print(f"Hit rate: {metrics.hit_rate:.1%}")
        print(f"Avg hit latency: {metrics.hit_latency_ms / max(metrics.hits, 1):.1f}ms")
        print(f"Avg miss latency: {metrics.miss_latency_ms / max(metrics.misses, 1):.1f}ms")
