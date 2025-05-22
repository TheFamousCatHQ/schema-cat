import asyncio
import logging
import random
from functools import wraps
from typing import TypeVar, Callable, Any, Optional, Type, List, Union

logger = logging.getLogger("schema_cat")

T = TypeVar('T')

# Common exceptions that should trigger a retry
DEFAULT_RETRY_EXCEPTIONS = [
    # Network errors
    "ConnectionError",
    "TimeoutError",
    "httpx.ConnectError",
    "httpx.ReadTimeout",
    "httpx.WriteTimeout",
    "httpx.ConnectTimeout",
    "httpx.PoolTimeout",

    # Rate limiting and server errors
    "openai.RateLimitError",
    "openai.APIStatusError",
    "anthropic.RateLimitError",
    "anthropic.APIStatusError",
    "anthropic.APIError",
]

async def retry_with_exponential_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Optional[List[Union[str, Type[Exception]]]] = None,
    **kwargs: Any
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases with each retry
        jitter: Whether to add random jitter to the delay
        retry_exceptions: List of exception types or exception type names to retry on.
                         If None, uses DEFAULT_RETRY_EXCEPTIONS.
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        The last exception raised by the function if all retries fail
    """
    if retry_exceptions is None:
        retry_exceptions = DEFAULT_RETRY_EXCEPTIONS

    # Convert string exception names to actual exception types when possible
    exception_types = []
    for exc in retry_exceptions:
        if isinstance(exc, str):
            # Try to find the exception type in the global namespace
            parts = exc.split('.')
            if len(parts) > 1:
                # Try to import the module and get the exception
                try:
                    module_name = '.'.join(parts[:-1])
                    exc_name = parts[-1]
                    module = __import__(module_name, fromlist=[exc_name])
                    exception_types.append(getattr(module, exc_name))
                except (ImportError, AttributeError):
                    # If we can't import it, we'll check exception names at runtime
                    pass
            # Keep the string version for runtime checking
            exception_types.append(exc)
        else:
            # It's already an exception type
            exception_types.append(exc)

    delay = initial_delay
    last_exception = None

    for retry_count in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # Check if this exception should trigger a retry
            should_retry = False
            for exc_type in exception_types:
                if isinstance(exc_type, type) and isinstance(e, exc_type):
                    should_retry = True
                    break
                elif isinstance(exc_type, str) and exc_type == type(e).__name__:
                    should_retry = True
                    break
                elif isinstance(exc_type, str) and exc_type == f"{type(e).__module__}.{type(e).__name__}":
                    should_retry = True
                    break

            if not should_retry or retry_count >= max_retries:
                logger.error(f"Failed after {retry_count + 1} attempts: {str(e)}")
                raise

            # Calculate the next delay with optional jitter
            if jitter:
                jitter_amount = random.uniform(0.0, 0.1 * delay)
                actual_delay = delay + jitter_amount
            else:
                actual_delay = delay

            logger.warning(
                f"Attempt {retry_count + 1}/{max_retries} failed with error: {str(e)}. "
                f"Retrying in {actual_delay:.2f} seconds..."
            )

            await asyncio.sleep(actual_delay)
            delay = min(delay * backoff_factor, max_delay)

    # This should never happen, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in retry logic")

def with_retry(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Optional[List[Union[str, Type[Exception]]]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry an async function with exponential backoff.

    Args:
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases with each retry
        jitter: Whether to add random jitter to the delay
        retry_exceptions: List of exception types or exception type names to retry on.
                         If None, uses DEFAULT_RETRY_EXCEPTIONS.

    Returns:
        A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Use parameters from kwargs if provided, otherwise use decorator defaults
            retry_kwargs = {
                'max_retries': kwargs.pop('max_retries', max_retries),
                'initial_delay': kwargs.pop('initial_delay', initial_delay),
                'max_delay': kwargs.pop('max_delay', max_delay),
                'backoff_factor': kwargs.pop('backoff_factor', backoff_factor),
                'jitter': kwargs.pop('jitter', jitter),
                'retry_exceptions': kwargs.pop('retry_exceptions', retry_exceptions),
            }

            return await retry_with_exponential_backoff(
                func,
                *args,
                **retry_kwargs,
                **kwargs
            )
        return wrapper
    return decorator
