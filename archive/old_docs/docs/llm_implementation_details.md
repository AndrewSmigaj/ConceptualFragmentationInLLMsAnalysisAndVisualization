# LLM Integration Technical Details

This document provides technical implementation details for the LLM integration in the Conceptual Fragmentation Analysis framework. It's intended for developers who want to understand the architecture, extend functionality, or modify the implementation.

## Architecture Overview

The LLM integration follows a layered architecture:

```
Dashboard UI (llm_tab.py)
       ↑
High-level Analysis API (analysis.py)
       ↑
Factory & Client Management (factory.py)
       ↑
Provider-specific Clients (openai_client.py, claude.py, etc.)
       ↑
Base LLM Client (client.py)
```

## Core Components

### Base LLM Client (`client.py`)

The `BaseLLMClient` is an abstract base class that defines the contract for all provider implementations:

```python
class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from the LLM based on the provided prompt."""
        pass
```

Key features:
- Abstract interface for generating text
- Standardized parameters (prompt, temperature, max_tokens)
- Returns an `LLMResponse` object with text and metadata
- Provider-specific parameters can be passed through `**kwargs`

### Provider-specific Implementations

Each supported provider has its own client implementation:

- **OpenAI** (`openai_client.py`): Uses the OpenAI API Python library
- **Claude** (`claude.py`): Uses the Anthropic API Python library
- **Grok** (`grok.py`): Uses a custom HTTP client for the Meta AI API
- **Gemini** (`gemini.py`): Uses the Google AI Python SDK

Each client is responsible for:
- Converting the standard parameters to provider-specific ones
- Handling provider-specific error cases
- Managing any provider-specific rate limiting or quotas
- Monitoring token usage

### Factory Pattern (`factory.py`)

The `LLMClientFactory` implements a factory pattern to create the appropriate client:

```python
@classmethod
def create_client(
    cls,
    provider: str,
    api_key: Optional[str] = None,
    model: str = "default",
    config: Optional[Dict[str, Any]] = None
) -> BaseLLMClient:
    """Create an LLM client for the specified provider."""
    pass
```

Key features:
- Dynamic client instantiation based on provider name
- Automatic API key resolution (from parameters or environment)
- Dynamic client registration through the `register_clients()` function
- Normalization of provider names (e.g., "gpt" → OpenAIClient)

### High-level Analysis API (`analysis.py`)

The `ClusterAnalysis` class provides high-level functions for neural network interpretation:

```python
class ClusterAnalysis:
    def __init__(
        self,
        provider: str = "grok",
        model: str = "default",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        # ...other parameters...
    ):
        """Initialize the ClusterAnalysis."""
        pass
        
    async def label_cluster(self, cluster_centroid: np.ndarray, ...): pass
    async def label_clusters(self, cluster_centroids: Dict[str, np.ndarray], ...): pass
    async def generate_path_narrative(self, path: List[str], ...): pass
    async def generate_path_narratives(self, paths: Dict[int, List[str]], ...): pass
```

Key features:
- High-level functions for cluster labeling and path narratives
- Support for both synchronous and asynchronous operations
- Integrated caching system
- Batch processing for improved performance
- Prompt optimization

### Response Object (`responses.py`)

The `LLMResponse` class provides a standardized structure for responses:

```python
class LLMResponse:
    def __init__(
        self,
        text: str,
        model: str = "unknown",
        provider: str = "unknown",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        raw_response: Any = None,
        **kwargs
    ):
        """Initialize an LLM response object."""
        pass
```

Key features:
- Standard fields for all providers
- Token usage tracking 
- Access to provider-specific raw response
- Additional metadata through kwargs

### Caching System (`cache_manager.py`)

The `CacheManager` handles caching of LLM responses:

```python
class CacheManager:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
        memory_only: bool = False,
        save_interval: int = 10
    ):
        """Initialize the cache manager."""
        pass
        
    def get(self, prompt: str, **kwargs) -> Optional[LLMResponse]: pass
    def store(self, prompt: str, response: LLMResponse, **kwargs) -> None: pass
    def clear(self, force_save: bool = True) -> None: pass
```

Key features:
- In-memory caching with disk persistence
- Cache key generation from prompt and parameters
- Time-based expiration (TTL)
- Efficient disk I/O with batched writes
- Memory-only mode for ephemeral caching

### Batch Processing (`batch_processor.py`)

The batch processor enables concurrent API requests for better performance:

```python
async def batch_generate_labels(
    analyzer: ClusterAnalysis,
    items: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
    max_concurrency: int = 5
) -> Dict[str, str]:
    """Generate labels for multiple clusters concurrently."""
    pass
    
async def batch_generate_narratives(
    analyzer: ClusterAnalysis,
    items: Dict[int, Dict[str, Any]],
    max_concurrency: int = 5
) -> Dict[int, str]:
    """Generate narratives for multiple paths concurrently."""
    pass
```

Key features:
- Concurrent processing with bounded concurrency
- Error handling for individual items
- Progress tracking and logging

### Prompt Optimization (`prompt_optimizer.py`)

The prompt optimizer improves prompts for better results:

```python
def optimize_cluster_label_prompt(prompt: str, level: int = 1) -> str:
    """Optimize the cluster labeling prompt."""
    pass
    
def optimize_path_narrative_prompt(prompt: str, level: int = 1) -> str:
    """Optimize the path narrative prompt."""
    pass
```

Key features:
- Different optimization levels (1-3)
- Token efficiency improvements
- Context clarity enhancements

## Dashboard Integration (`llm_tab.py`)

The dashboard integration provides a user interface:

```python
def create_llm_tab():
    """Create the layout for the LLM Integration tab."""
    pass
    
def register_llm_callbacks(app):
    """Register callbacks for the LLM Integration tab."""
    pass
```

Key features:
- Dash-based UI components
- Reactive callbacks for UI updates
- Integration with dashboard data flow
- Results display formatting

## Threading Model

The LLM integration supports both synchronous and asynchronous operations:

### Asynchronous Operations

- Core methods like `generate()`, `label_clusters()`, and `generate_path_narratives()` are async
- Batch processing is implemented with `asyncio.gather()` for concurrent execution
- Semaphores are used to limit concurrency (`asyncio.Semaphore`)

### Synchronous Wrappers

- Methods with `_sync` suffix (e.g., `label_clusters_sync()`) wrap async methods
- They create and manage an event loop if needed
- They're used by the dashboard, which operates synchronously

## Error Handling

The LLM integration includes comprehensive error handling:

- Provider-specific errors are caught and translated to standard exceptions
- Network errors are retried with exponential backoff
- API rate limits are respected through delaying or queuing
- Validation errors prevent invalid operations
- Detailed error messages help with debugging

## Data Flow

A typical data flow for generating cluster labels:

1. User selects parameters in the dashboard
2. Dashboard calls `ClusterAnalysis.label_clusters_sync()`
3. The synchronous wrapper sets up an event loop
4. The asynchronous `label_clusters()` is called
5. For each cluster:
   a. A prompt is generated based on centroid values
   b. The cache is checked for existing responses
   c. If not cached, the appropriate LLM client is used
   d. The response is cached for future use
6. Results are collected and returned to the dashboard
7. The dashboard updates the UI with the results

## Extension Points

### Adding a New Provider

To add a new LLM provider:

1. Create a new client class that extends `BaseLLMClient`
2. Implement the `generate()` method using the provider's API
3. Add the provider to the factory registration in `register_clients()`
4. Update environment variable names in `ENV_API_KEY_MAP`
5. Add the provider to the dashboard UI options

Example of a minimal client implementation:

```python
class NewProviderClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "default",
        **kwargs
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model if model != "default" else "new-provider-default-model"
        
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        # Provider-specific implementation
        # ...
        
        return LLMResponse(
            text=response_text,
            model=self.model,
            provider="new-provider",
            # ... other parameters ...
        )
```

### Customizing Prompts

To customize the prompts used:

1. Modify the appropriate method in `ClusterAnalysis`:
   - `label_cluster()` for cluster labeling
   - `generate_path_narrative()` for path narratives
2. Or enhance the prompt optimization in `prompt_optimizer.py`

### Adding New Analysis Types

To add a new type of LLM-based analysis:

1. Add a new method to `ClusterAnalysis`
2. Create corresponding UI elements in `create_llm_tab()`
3. Add a new callback in `register_llm_callbacks()`
4. Create a display component for the results

## Performance Considerations

- **Token Usage**: Monitor token usage to control costs
- **Caching**: Use caching to avoid redundant API calls
- **Concurrency**: Adjust `max_concurrency` based on provider rate limits
- **Batch Size**: Process data in appropriately sized batches
- **Prompt Efficiency**: Optimize prompts to reduce token count

## Security Considerations

- **API Keys**: Never hardcode API keys; use environment variables or secure storage
- **Data Privacy**: Be mindful of any sensitive data in prompts
- **Rate Limiting**: Implement proper rate limiting to prevent abuse
- **Validation**: Validate all inputs before sending to external APIs
- **Error Exposure**: Avoid exposing detailed error messages to users