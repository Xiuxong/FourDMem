"""Auto-Plugin Generator — Self-Growing Cognitive Organs

Detects retrieval pain points from failed queries and generates
specialized Python plugins to handle them.

Implements T-10.8: the system can "grow" new retrieval capabilities
by generating, validating, and hot-installing Python plugins.

Pain point examples:
- Complex regex patterns that fulltext can't handle
- Domain-specific terminology requiring custom tokenization
- Multi-language queries needing special processing
"""

import os
import re
from typing import Any


try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)




class AutoPluginGenerator:
    """Detects retrieval pain points and generates specialized plugins.

    Usage:
        generator = AutoPluginGenerator()
        pain_point = generator.detect_pain_point(failed_queries)
        if pain_point:
            code = generator.generate_plugin(pain_point)
            generator.validate_and_install(code, pain_point)
    """

    PAIN_PATTERNS = {
        "regex": r"[\[\]{}()\\|+*?^$\.]",
        "code_snippet": r"(def |class |import |from \w+ import|```)",
        "math_formula": r"[∑∫∂√∞≈≠≤≥±×÷]",
        "multi_lang": r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]",
    }

    def __init__(self, plugin_dir: str = "data/plugins"):
        self.plugin_dir = plugin_dir

    def detect_pain_point(self, failed_queries: list[str]) -> str | None:
        """Analyze failed queries to identify a pain point category.

        Args:
            failed_queries: Queries that returned no/few results.

        Returns:
            Pain point category name, or None if no pattern detected.
        """
        if not failed_queries:
            return None

        # Count matches per pattern
        scores: dict[str, int] = {}
        for category, pattern in self.PAIN_PATTERNS.items():
            count = sum(1 for q in failed_queries if re.search(pattern, q))
            if count > 0:
                scores[category] = count

        if not scores:
            return None

        # Return the most common pain point
        return max(scores, key=scores.get)

    def generate_plugin(self, pain_point: str) -> dict:
        """Generate a cognition task for the Agent to write a plugin.

        The Agent uses its LLM to generate Python plugin code that handles
        the detected pain point. Returns a structured task, not auto-generated code.

        Args:
            pain_point: The pain point category (e.g. "regex", "code_snippet").

        Returns:
            Dict with status, type, pain_point, and instruction.
        """
        return {
            "status": "cognition_task",
            "type": "generate_plugin",
            "pain_point": pain_point,
            "instruction": (
                f"Generate a Python plugin for the '{pain_point}' retrieval pain point. "
                "The plugin must have a preprocess_query(query: str) -> str function. "
                "Write safe, validated code. Call validate_and_install() to install."
            ),
            "plugin_path": os.path.join(self.plugin_dir, f"plugin_{pain_point}.py"),
        }

    def validate_and_install(self, code: str, pain_point: str) -> bool:
        """Validate and install a generated plugin.

        Args:
            code: The plugin Python source code.
            pain_point: The pain point category.

        Returns:
            True if installed successfully.
        """
        # Basic safety validation
        dangerous_patterns = [
            "import os", "import subprocess", "import shutil",
            "exec(", "eval(", "__import__", "open(",
            "os.system", "os.popen",
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                logger.error(f"Plugin rejected: contains dangerous pattern '{pattern}'")
                return False

        # Write to plugin directory
        os.makedirs(self.plugin_dir, exist_ok=True)
        plugin_path = os.path.join(self.plugin_dir, f"plugin_{pain_point}.py")

        try:
            with open(plugin_path, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info(f"Plugin installed: {plugin_path}")
            return True
        except Exception as e:
            logger.error(f"Plugin installation failed: {e}")
            return False

    def detect_and_generate(self, failed_queries: list[str]) -> dict:
        """Detect pain point and return a cognition task for the Agent.

        The Agent processes the returned task with its LLM to generate
        actual plugin code, then calls validate_and_install() to install.

        Returns:
            Dict with status and cognition_task if a pain point is detected.
        """
        pain_point = self.detect_pain_point(failed_queries)
        if not pain_point:
            return {"status": "no_pain_point"}

        return self.generate_plugin(pain_point)

    def load_installed_plugins(self) -> dict[str, Any]:
        """Load all installed plugins from the plugin directory.

        Returns:
            Dict of plugin_name -> plugin module.
        """
        import importlib.util

        plugins = {}
        if not os.path.exists(self.plugin_dir):
            return plugins

        for filename in os.listdir(self.plugin_dir):
            if filename.startswith("plugin_") and filename.endswith(".py"):
                plugin_name = filename[:-3]
                filepath = os.path.join(self.plugin_dir, filename)

                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, filepath)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        plugins[plugin_name] = module
                except Exception as e:
                    logger.warning(f"Failed to load plugin {filename}: {e}")

        return plugins

    def preprocess_query(self, query: str, plugins: dict[str, Any] | None = None) -> str:
        """Apply all installed plugin preprocessors to a query.

        Args:
            query: Original query string.
            plugins: Pre-loaded plugins dict (optional, loads if None).

        Returns:
            Preprocessed query string.
        """
        if plugins is None:
            plugins = self.load_installed_plugins()

        processed = query
        for name, plugin in plugins.items():
            try:
                if hasattr(plugin, "preprocess_query"):
                    processed = plugin.preprocess_query(processed)
            except Exception as e:
                logger.warning(f"Plugin {name} preprocessing failed: {e}")

        return processed
