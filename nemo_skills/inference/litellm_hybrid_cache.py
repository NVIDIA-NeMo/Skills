# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Hybrid Cache implementation that keeps data in memory and periodically saves to a single file on disk.

This avoids the issue with disk cache creating too many files while still persisting cache across runs.

Also includes StableLiteLLMCache which fixes litellm's cache key generation to be order-independent.
"""

import atexit
import json
import os
import pickle
import threading
from pathlib import Path
from typing import List, Optional

import litellm
from litellm.caching.caching import Cache as LiteLLMCache


class HybridCache:
    def __init__(
        self,
        cache_file_path: str,
        save_interval_seconds: float = 300.0,  # 5 minutes
    ):
        self.cache_file_path = cache_file_path
        self.save_interval_seconds = save_interval_seconds

        self.cache_dict: dict = {}
        self._lock = threading.RLock()
        self._dirty = False  # Track if cache has been modified since last save
        self._stop_event = threading.Event()
        self._save_thread: Optional[threading.Thread] = None

        self._load_from_disk()
        self._start_background_save_thread()

        atexit.register(self._shutdown)

    def _check_no_ttl(self, **kwargs):
        """Raise error if TTL is provided since TTL is not supported."""
        if kwargs.get("ttl") is not None:
            raise ValueError("TTL is not supported by HybridCache")

    def _load_from_disk(self):
        """Load cache from disk if the file exists."""
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "rb") as f:
                data = pickle.load(f)
                self.cache_dict = data["cache_dict"]

    def _save_to_disk(self):
        """Save cache to disk."""
        with self._lock:
            if not self._dirty:
                return
            data = {
                "cache_dict": self.cache_dict.copy(),
            }
            self._dirty = False

        temp_path = self.cache_file_path + ".tmp"
        Path(self.cache_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, self.cache_file_path)

    def _start_background_save_thread(self):
        """Start a background thread that periodically saves to disk."""

        def save_loop():
            while not self._stop_event.wait(timeout=self.save_interval_seconds):
                self._save_to_disk()

        self._save_thread = threading.Thread(target=save_loop, daemon=True)
        self._save_thread.start()

    def _shutdown(self):
        """Shutdown the background thread and save cache."""
        self._stop_event.set()
        if self._save_thread is not None:
            self._save_thread.join(timeout=5.0)
        self._save_to_disk()

    def set_cache(self, key, value, **kwargs):
        """Set a value in the cache."""
        self._check_no_ttl(**kwargs)
        with self._lock:
            self.cache_dict[key] = value
            self._dirty = True

    async def async_set_cache(self, key, value, **kwargs):
        """Async set - delegates to sync implementation since we're using in-memory."""
        self.set_cache(key=key, value=value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        """Set multiple cache entries."""
        for cache_key, cache_value in cache_list:
            self.set_cache(key=cache_key, value=cache_value, **kwargs)

    def get_cache(self, key, **kwargs):
        """Get a value from the cache."""
        with self._lock:
            if key not in self.cache_dict:
                return None
            cached_response = self.cache_dict[key]
            if isinstance(cached_response, str):
                try:
                    cached_response = json.loads(cached_response)
                except json.JSONDecodeError:
                    pass
            return cached_response

    async def async_get_cache(self, key, **kwargs):
        """Async get - delegates to sync implementation."""
        return self.get_cache(key=key, **kwargs)

    def batch_get_cache(self, keys: list, **kwargs):
        """Get multiple values from cache."""
        return [self.get_cache(key=k, **kwargs) for k in keys]

    async def async_batch_get_cache(self, keys: list, **kwargs):
        """Async batch get."""
        return self.batch_get_cache(keys=keys, **kwargs)

    def increment_cache(self, key, value: int, **kwargs) -> int:
        """Increment a cache value."""
        with self._lock:
            init_value = self.get_cache(key=key) or 0
            new_value = init_value + value
            self.set_cache(key, new_value, **kwargs)
            return new_value

    async def async_increment(self, key, value: float, **kwargs) -> float:
        """Async increment."""
        return self.increment_cache(key, int(value), **kwargs)

    def flush_cache(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache_dict.clear()
            self._dirty = True

    def delete_cache(self, key):
        """Delete a specific key from cache."""
        with self._lock:
            self.cache_dict.pop(key, None)
            self._dirty = True

    async def disconnect(self):
        """Disconnect and save cache to disk."""
        self._shutdown()

    async def async_set_cache_sadd(self, key, value: List):
        """Add values to a set."""
        with self._lock:
            init_value = self.get_cache(key=key) or set()
            for val in value:
                init_value.add(val)
            self.set_cache(key, init_value)
            return value

    def force_save(self):
        """Force an immediate save to disk."""
        self._dirty = True
        self._save_to_disk()


class StableLiteLLMCache(LiteLLMCache):
    """
    A litellm Cache subclass that generates order-independent cache keys.

    The default litellm cache key generation iterates through kwargs in order,
    which means the same request with different parameter ordering produces
    different cache keys. This class fixes that by sorting kwargs before iteration.
    """

    def __init__(self, cache_file_path: str, save_interval_seconds: float = 300.0, **kwargs):
        super().__init__(type="local", **kwargs)
        self.cache = HybridCache(
            cache_file_path=cache_file_path,
            save_interval_seconds=save_interval_seconds,
        )

    def _stable_str(self, value) -> str:
        """Convert value to string deterministically (handles nested dicts/lists)."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, default=str)
        return str(value)

    def get_cache_key(self, **kwargs) -> str:
        """
        Get the cache key for the given arguments.
        Same as parent but with sorted iteration for deterministic keys.
        """
        from litellm.litellm_core_utils.model_param_helper import ModelParamHelper
        from litellm.types.utils import all_litellm_params

        cache_key = ""

        preset_cache_key = self._get_preset_cache_key_from_kwargs(**kwargs)
        if preset_cache_key is not None:
            return preset_cache_key

        combined_kwargs = ModelParamHelper._get_all_llm_api_params()
        litellm_param_kwargs = all_litellm_params

        # FIX: Sort kwargs for deterministic cache key generation
        for param in sorted(kwargs.keys()):
            if param in combined_kwargs:
                param_value = self._get_param_value(param, kwargs)
                if param_value is not None:
                    cache_key += f"{str(param)}: {self._stable_str(param_value)}"
            elif param not in litellm_param_kwargs:
                if litellm.enable_caching_on_provider_specific_optional_params is True:
                    if kwargs[param] is None:
                        continue
                    param_value = kwargs[param]
                    cache_key += f"{str(param)}: {self._stable_str(param_value)}"

        hashed_cache_key = self._get_hashed_cache_key(cache_key)
        hashed_cache_key = self._add_namespace_to_cache_key(hashed_cache_key, **kwargs)
        self._set_preset_cache_key_in_kwargs(preset_cache_key=hashed_cache_key, **kwargs)
        return hashed_cache_key
