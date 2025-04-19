# -*- coding: utf-8 -*-
"""
Gemini Chat Agent v9.8: Codespaces Ready, Clean Console, Increased Context

Core Features:
- Model: gemini/gemini-2.0-flash
- Reads API Keys from .env file.
- Console Output: Shows only chat flow (Thinking, Tool Use, Response) & critical errors.
- Tools: Weather, Search, Firecrawl, GitHub, CodeExec(Ack), DateTime, VectorSearch
- Increased Context Window (1M tokens).
- Vector DB (in-memory), Conversation Memory Pruning.
- Streaming Responses.
- Structured Logging (Detailed logs -> file).
- File Handling.
"""

# --- Installations ---
import subprocess
import sys

def install_packages():
    packages = [
        "litellm",
        "python-dotenv",
        "requests",
        "duckduckgo-search",
        "firecrawl-py",
        "sentence-transformers",
        "numpy",
        "matplotlib",
        "PyGithub"
    ]
    try:
        print("Checking and installing necessary libraries...") # Console print for setup
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
        print("Libraries installed/verified.") # Console print for setup
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}", file=sys.stderr) # Console print for setup error
        sys.exit(1)
    except FileNotFoundError:
         print("Error: 'pip' command not found. Please ensure Python environment is set up correctly.", file=sys.stderr) # Console print for setup error
         sys.exit(1)

install_packages()

# --- Imports ---
import litellm, os, base64, json, mimetypes, getpass, requests, traceback, logging, numpy as np, time, io
from pathlib import Path
from duckduckgo_search import DDGS
from firecrawl import FirecrawlApp
from datetime import datetime
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
from github import Github, GithubException
import tempfile
import shutil
from dotenv import load_dotenv # For .env file

# --- Load Environment Variables ---
load_dotenv()
print("Attempted to load API keys from .env file.") # Console print for setup

# --- Setup Logging (Clean Console) ---
def setup_logging():
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("gemini_agent")
    logger.setLevel(logging.INFO)
    # File Handler - Detailed logs
    file_handler = logging.FileHandler("gemini_agent_v9.8.log")
    file_handler.setFormatter(log_formatter); file_handler.setLevel(logging.INFO)
    # Stream Handler - Only critical info to console
    stream_handler = logging.StreamHandler(); stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.WARNING) # Console only gets WARNING+
    logger.addHandler(file_handler); logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
logger = setup_logging()
logger.info("--- Starting Gemini Chat Agent v9.8 (Codespaces Ready) ---") # To file only

# --- Custom Exceptions ---
class AgentException(Exception): pass
class ToolExecutionError(AgentException): pass
class APIKeyError(AgentException): pass
class VectorDBError(AgentException): pass
class GitHubToolError(ToolExecutionError): pass

# --- API Key Setup (from Environment) ---
logger.info("Setting up API Keys from environment...") # To file only
def get_required_key(env_var):
    key = os.environ.get(env_var)
    if not key:
        logger.error(f"CRITICAL: Environment variable {env_var} not set. Please set it in .env or system environment.")
        raise APIKeyError(f"Missing required API key: {env_var}")
    logger.info(f"Found required key: {env_var}") # To file only
    return key

def get_optional_key(env_var):
    key = os.environ.get(env_var)
    if not key:
        logger.warning(f"Optional environment variable {env_var} not set. Related tools may not function.") # Console WARN
    else:
        logger.info(f"Found optional key: {env_var}") # To file only
    return key

try:
    gemini_api_key = get_required_key("GEMINI_API_KEY")
    openweathermap_api_key = get_optional_key("OPENWEATHERMAP_API_KEY")
    firecrawl_api_key = get_optional_key("FIRECRAWL_API_KEY")
    github_api_key = get_optional_key("GITHUB_API_KEY")
    stability_api_key = get_optional_key("STABILITY_API_KEY")
except APIKeyError as e:
    print(f"Error: {e}. Please ensure it's set in your .env file or environment.", file=sys.stderr) # Console Error
    sys.exit(1)
except Exception as e:
     logger.critical(f"Unexpected error during API key setup: {e}", exc_info=True) # File CRITICAL
     print(f"Unexpected critical error during API key setup: {e}", file=sys.stderr) # Console CRITICAL
     sys.exit(1)
logger.info("API Keys configured.") # To file only


# --- Memory Management ---
# (Unchanged - uses logger internally)
class ConversationMemory:
    CHARS_PER_TOKEN_ESTIMATE = 4
    def __init__(self, max_tokens=1_000_000, system_message=None): # Increased limit
        self.messages = []; self.system_message = system_message
        if system_message: self.messages.append(system_message); self.token_count = self._estimate_tokens(system_message)
        else: self.token_count = 0
        self.max_tokens = max_tokens; logger.info(f"Memory init: max_tokens={max_tokens}")
    def _estimate_tokens(self, message): return len(json.dumps(message)) // self.CHARS_PER_TOKEN_ESTIMATE
    def add_message(self, message):
        est_tokens = self._estimate_tokens(message)
        while self.token_count + est_tokens > self.max_tokens and len(self.messages) > (1 if self.system_message else 0): self._prune_history()
        if self.token_count + est_tokens > self.max_tokens: logger.warning(f"Msg ({est_tokens} tk) too large. Skipping."); return False
        self.messages.append(message); self.token_count += est_tokens; logger.debug(f"Msg added. Role: {message.get('role')}, Tokens: {self.token_count}"); return True
    def _prune_history(self):
        if len(self.messages) <= (1 if self.system_message else 0): logger.warning("Cannot prune."); return
        idx = 1 if self.system_message else 0
        if idx < len(self.messages):
            removed = self.messages.pop(idx); removed_tk = self._estimate_tokens(removed)
            self.token_count -= removed_tk; logger.info(f"Pruned msg (Role: {removed.get('role')}). Tokens: {self.token_count}")
        else: logger.warning("Pruning failed: No non-system msgs.")
    def get_messages(self): return self.messages
    def get_last_user_message_content(self):
        for msg in reversed(self.messages):
            if msg.get("role") == "user": return msg.get("content")
        return None

# --- Vector Database for Semantic Memory ---
# (Unchanged - uses logger internally)
class VectorDB:
    def __init__(self):
        self.model = None; self.texts = []; self.metadata = []; self.embeddings = None; self.initialized = False
        try: from sentence_transformers import SentenceTransformer; self.model = SentenceTransformer('all-MiniLM-L6-v2'); self.initialized = True; logger.info("Vector DB init success")
        except ImportError: logger.error("sentence-transformers not installed. VDB disabled.")
        except Exception as e: logger.error(f"SentenceTransformer load failed: {e}. VDB disabled.")
    def add(self, text, metadata=None):
        if not self.is_ready(): logger.warning("VDB add skipped: Not initialized."); return False
        if not text or not isinstance(text, str): logger.warning(f"VDB add skipped: Invalid text."); return False
        try:
            embedding = self.model.encode(text)
            if self.embeddings is None: self.embeddings = np.array([embedding])
            else:
                if not isinstance(self.embeddings, np.ndarray): logger.error("VDB state error."); return False
                self.embeddings = np.vstack([self.embeddings, embedding])
            self.texts.append(text); self.metadata.append(metadata or {})
            logger.debug(f"Added VDB entry: {text[:50]}... Shape: {self.embeddings.shape if self.embeddings is not None else 'None'}")
            return True
        except Exception as e: logger.error(f"VDB add error: {e}", exc_info=True); return False
    def search(self, query, top_k=3):
        if not self.is_ready(): logger.error("VDB search fail: Not initialized."); raise VectorDBError("VDB not initialized")
        if self.embeddings is None or self.embeddings.size == 0: logger.warning("VDB empty."); return []
        try:
            q_emb = self.model.encode(query); embs_2d = self.embeddings if self.embeddings.ndim == 2 else self.embeddings.reshape(-1, self.model.get_sentence_embedding_dimension())
            norms_e = np.linalg.norm(embs_2d, axis=1); norm_q = np.linalg.norm(q_emb)
            if norm_q == 0: logger.warning("Query norm zero."); return []
            valid_idx = np.where(norms_e > 1e-8)[0]
            if len(valid_idx) == 0: logger.warning("No valid embeddings."); return []
            valid_embs = embs_2d[valid_idx]; valid_norms = norms_e[valid_idx]
            sims = np.dot(valid_embs, q_emb) / (valid_norms * norm_q)
            count = min(top_k, len(sims));
            if count <= 0: return []
            rel_top_idx = np.argsort(sims)[::-1][:count]; orig_top_idx = valid_idx[rel_top_idx]
            results = []
            for i in orig_top_idx:
                 if i < len(self.texts): results.append({ "text": self.texts[i], "similarity": float(sims[rel_top_idx[np.where(orig_top_idx == i)[0][0]]]), "metadata": self.metadata[i] })
                 else: logger.warning(f"Search index {i} out of bounds.")
            logger.info(f"VDB search '{query[:30]}...' returned {len(results)}.")
            return results
        except Exception as e: logger.error(f"VDB search error: {e}", exc_info=True); raise VectorDBError(f"VDB search failed: {e}")
    def is_ready(self): return self.initialized and self.model is not None
vector_db = VectorDB()

# --- Tool Implementation (Object-Oriented) ---
# (Base Tool class and specific tool implementations unchanged - use logger internally)
class Tool:
    def __init__(self, name, description, parameters=None, required=None):
        self.name = name; self.description = description
        if parameters and not isinstance(parameters, dict): raise ValueError("Params must be dict.")
        self.parameters = parameters or {"type": "object", "properties": {}}
        if required and not isinstance(required, list): raise ValueError("Required must be list.")
        self.required = required or []
        if self.required: self.parameters["required"] = self.required
    def get_schema(self): return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}
    def validate_args(self, args):
        if not isinstance(args, dict): raise ToolExecutionError("Args must be dict.")
        missing = [p for p in self.required if p not in args or args[p] is None]
        if missing: raise ToolExecutionError(f"Missing required: {', '.join(missing)}")
        return True
    def execute(self, **kwargs): raise NotImplementedError("Subclass must implement")

class WeatherTool(Tool):
    def __init__(self): super().__init__(name="get_current_weather", description="Retrieves real-time weather conditions for a specific city.", parameters={"type": "object", "properties": { "location": {"type": "string", "description": "City name."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temp unit."}}}, required=["location"])
    def execute(self, **kwargs):
        self.validate_args(kwargs); l = kwargs.get("location"); u = kwargs.get("unit", "celsius")
        if not openweathermap_api_key: raise ToolExecutionError("Weather key missing.")
        url="http://api.openweathermap.org/data/2.5/weather";unts="metric" if u=="celsius" else "imperial";sym="°C" if u=="celsius" else "°F";p={"q":l,"appid":openweathermap_api_key,"units":unts}
        retries=3;delay=1
        for attempt in range(retries):
            try:
                r=requests.get(url, params=p, timeout=10);r.raise_for_status();data=r.json()
                if data.get("cod") != 200: raise ToolExecutionError(f"Weather API Error: {data.get('message', 'Unknown')}")
                m=data.get("main",{});w=data.get("weather",[{}]);d=w[0].get('description',"");t=m.get('temp');f=m.get('feels_like');h=m.get('humidity')
                res=f"Weather in {data.get('name', l)}: {d}, Temp: {t}{sym} (feels like {f}{sym}), Humidity: {h}%"
                if vector_db.is_ready(): vector_db.add(f"Weather: {l}({u}): {res}", {"type": "weather", "location": l, "time": datetime.now().isoformat()})
                return res
            except requests.exceptions.Timeout: logger.warning(f"Weather timeout {l} (try {attempt+1}). Retrying...")
            except requests.exceptions.HTTPError as e:
                 if e.response.status_code == 404: raise ToolExecutionError(f"City '{l}' not found.")
                 elif e.response.status_code == 401: raise ToolExecutionError("Invalid Weather API key.")
                 else: logger.error(f"Weather HTTP error: {e}"); raise ToolExecutionError(f"HTTP error {e.response.status_code}")
            except requests.exceptions.RequestException as e: logger.error(f"Weather network error: {e}"); raise ToolExecutionError(f"Network error: {e}")
            except Exception as e: logger.error(f"Unexpected weather error: {e}"); raise ToolExecutionError(f"Unexpected error: {e}")
            time.sleep(delay); delay *= 2
        raise ToolExecutionError(f"Weather fetch failed after {retries} attempts.")
class SearchTool(Tool):
    def __init__(self): super().__init__(name="perform_web_search", description="General web search for facts/current info.", parameters={"type": "object", "properties": {"query": {"type": "string", "description": "Search query."}}}, required=["query"])
    def execute(self, **kwargs):
        self.validate_args(kwargs); q = kwargs.get("query"); logger.info(f"Searching: {q}")
        try:
            with DDGS() as ddgs: results = list(ddgs.text(q, max_results=5))
            if not results: return f"No results for '{q}'."
            fmt = []
            for r in results:
                txt = f"Title: {r.get('title','N/A')}\nSnippet: {r.get('body','N/A')}\nURL: {r.get('href','N/A')}"
                fmt.append(txt)
                if vector_db.is_ready(): vector_db.add(f"Search snippet '{q}': {r.get('title', '')} - {r.get('body', '')}", {"type": "search_result", "url": r.get('href'), "query": q, "time": datetime.now().isoformat()})
            return f"Search results for '{q}':\n\n" + "\n\n---\n\n".join(fmt)
        except Exception as e: logger.error(f"Search error: {e}"); raise ToolExecutionError(f"Search failed: {e}")
class WebScraperTool(Tool):
    def __init__(self): super().__init__(name="scrape_website_for_llm", description="Fetches main content of a specific URL as Markdown.", parameters={"type": "object", "properties": {"url": {"type": "string", "description": "URL to scrape."}}}, required=["url"])
    def execute(self, **kwargs):
        self.validate_args(kwargs); url = kwargs.get("url"); logger.info(f"Scraping URL: {url}")
        if not firecrawl_api_key: raise ToolExecutionError("Firecrawl API key missing.")
        try:
            app = FirecrawlApp(api_key=firecrawl_api_key)
            # Using user-specified call structure
            scraped_data = app.scrape_url(url=url, params={'formats': ['markdown']})
            markdown_content = None
            if isinstance(scraped_data, dict): markdown_content = scraped_data.get('markdown')
            elif isinstance(scraped_data, str): markdown_content = scraped_data
            if markdown_content:
                logger.info(f"Scrape success: {url}")
                if vector_db.is_ready():
                    chunks = self._chunk_content(markdown_content); logger.info(f"Storing {len(chunks)} chunks from {url} in VDB.")
                    for i, chunk in enumerate(chunks): vector_db.add(chunk, {"type": "web_content", "url": url, "chunk": i+1, "total_chunks": len(chunks), "time": datetime.now().isoformat()})
                return markdown_content
            else:
                error_msg = scraped_data.get('error', 'Markdown content not found or scrape failed.') if isinstance(scraped_data, dict) else "Scrape returned empty/unexpected data."
                logger.warning(f"Scrape failed for {url}: {error_msg}"); raise ToolExecutionError(f"Scraping failed: {error_msg}")
        except requests.exceptions.HTTPError as e:
             logger.error(f"Scrape HTTP error: {e}"); msg = f"Status {e.response.status_code}"
             try: details = e.response.json(); msg += f". Details: {details.get('error', details.get('message', json.dumps(details)))}"
             except json.JSONDecodeError: msg += f". Response: {e.response.text}"
             raise ToolExecutionError(f"Firecrawl API request failed. {msg}")
        except Exception as e: logger.error(f"Scrape exception: {e}"); traceback.print_exc(); raise ToolExecutionError(f"Unexpected scrape error: {e}")
    def _chunk_content(self, content, max_chars=1500, overlap=100):
        if not isinstance(content, str) or not content: return []
        if len(content) <= max_chars: return [content]
        chunks = []; start = 0
        while start < len(content):
            end = min(start + max_chars, len(content)); chunks.append(content[start:end])
            start += max_chars - overlap;
            if start >= len(content): break; start = max(0, start)
        return [c for c in chunks if c]
class CodeExecutionTool(Tool): # Acknowledgment only version
    def __init__(self): super().__init__(name="code_execution", description="Acknowledges Python code requests (execution by Gemini).", parameters={"type": "object", "properties": {"code": {"type": "string", "description": "Python code."}}}, required=["code"])
    def execute(self, **kwargs):
        self.validate_args(kwargs); code = kwargs.get("code"); logger.info(f"Ack code execution: {code[:100]}...")
        return "Code execution request acknowledged. Output will follow if executed by the AI."
class DateTimeTool(Tool):
    def __init__(self): super().__init__( name="get_current_datetime", description="Returns current date and time.", parameters={"type": "object", "properties": {}})
    def execute(self, **kwargs): now = datetime.now(); fmt = now.strftime("%A, %d %B %Y, %H:%M:%S %Z"); return f"Current date and time: {fmt}"
class VectorSearchTool(Tool):
    def __init__(self): super().__init__(name="semantic_memory_search", description="Searches agent's long-term memory (VDB) for relevant info.", parameters={"type": "object", "properties": { "query": {"type": "string", "description": "Search query for memory."}, "results_count": {"type": "integer", "description": "Num results (default: 3)."}}}, required=["query"])
    def execute(self, **kwargs):
        self.validate_args(kwargs); q = kwargs.get("query"); c = kwargs.get("results_count", 3)
        if not vector_db or not vector_db.is_ready(): raise ToolExecutionError("Vector DB unavailable.")
        try:
            res = vector_db.search(q, top_k=c)
            if not res: return "No relevant info found in memory."
            fmt = [f"Memory {i+1} (Relevance: {r['similarity']:.2f}):\nMetadata: {r.get('metadata', {})}\nContent: {r['text']}" for i, r in enumerate(res)]
            return "Semantic Memory Search Results:\n\n" + "\n\n---\n\n".join(fmt)
        except VectorDBError as e: logger.error(f"VDB search failed: {e}"); raise ToolExecutionError(f"Error searching memory: {e}")
        except Exception as e: logger.error(f"Unexpected VDB search error: {e}"); traceback.print_exc(); raise ToolExecutionError(f"Unexpected error searching memory: {e}")
class GitHubTool(Tool): # As provided
    def __init__(self): super().__init__(name="github_operations", description="Manage GitHub repositories, files and operations.", parameters={ "type": "object", "properties": { "operation": { "type": "string", "enum": ["list_repos", "create_repo", "read_file", "write_file", "list_files", "clone_repo"], "description": "GitHub operation."}, "repo_name": { "type": "string", "description": "[owner/]repo name." }, "file_path": { "type": "string", "description": "Path to file." }, "file_content": { "type": "string", "description": "Content to write." }, "description": { "type": "string", "description": "Repo description." }, "private": { "type": "boolean", "description": "Make repo private (default: false)." }, "branch": { "type": "string", "description": "Branch name (default: main/master)." }, "commit_message": { "type": "string", "description": "Commit message." }, "path": { "type": "string", "description": "Directory path." } }, "required": ["operation"] }, required=["operation"] )
    def _get_repo(self, github, repo_name):
        if not repo_name: raise GitHubToolError("'repo_name' required.")
        try: return github.get_repo(repo_name)
        except GithubException as e:
            if e.status == 404: raise GitHubToolError(f"Repo '{repo_name}' not found/accessible.")
            else: raise GitHubToolError(f"Error accessing repo '{repo_name}': {e.status} - {e.data.get('message', str(e))}")
        except Exception as e: raise GitHubToolError(f"Error getting repo '{repo_name}': {str(e)}")
    def execute(self, **kwargs):
        self.validate_args(kwargs); operation = kwargs.get("operation"); logger.info(f"Executing GitHub op: {operation}")
        if not github_api_key: raise GitHubToolError("GitHub API key missing.")
        try:
            g = Github(github_api_key); user = g.get_user()
            if operation == "list_repos":
                repos = list(user.get_repos(affiliation='owner'));
                if not repos: return "No owned repositories found."
                repo_list = [f"- {r.full_name} ({'private' if r.private else 'public'})" for i, r in enumerate(repos[:30])]
                return f"Your repositories ({min(len(repos), 30)} shown):\n" + "\n".join(repo_list)
            elif operation == "create_repo":
                repo_name = kwargs.get("repo_name")
                if not repo_name or "/" in repo_name: raise GitHubToolError("Valid repo name (no owner) required.")
                desc = kwargs.get("description", ""); priv = kwargs.get("private", False); logger.info(f"Creating repo: {repo_name}")
                repo = user.create_repo(name=repo_name, description=desc, private=priv, auto_init=True)
                if vector_db.is_ready(): vector_db.add(f"Created GitHub repo: {repo.full_name}", {"type": "github_action", "action": "create_repo", "repo": repo.full_name, "time": datetime.now().isoformat()})
                return f"Repository '{repo.full_name}' created: {repo.html_url}"
            repo_name = kwargs.get("repo_name"); repo = self._get_repo(g, repo_name)
            if operation == "read_file":
                fp = kwargs.get("file_path"); br = kwargs.get("branch", repo.default_branch);
                if not fp: raise GitHubToolError("'file_path' required.")
                logger.info(f"Reading '{fp}' from '{repo.full_name}' branch '{br}'")
                cf = repo.get_contents(fp, ref=br); content = base64.b64decode(cf.content).decode('utf-8')
                return f"Content of '{fp}' in '{repo.full_name}':\n```\n{content}\n```"
            elif operation == "write_file":
                fp = kwargs.get("file_path"); fc = kwargs.get("file_content"); cm = kwargs.get("commit_message"); br = kwargs.get("branch", repo.default_branch)
                if not fp or fc is None or not cm: raise GitHubToolError("file_path, file_content, commit_message required.")
                logger.info(f"Writing '{fp}' in '{repo.full_name}' branch '{br}'")
                try: contents = repo.get_contents(fp, ref=br); commit = repo.update_file(contents.path, cm, fc, contents.sha, branch=br); action = "updated"
                except GithubException as e:
                    if e.status == 404: commit = repo.create_file(fp, cm, fc, branch=br); action = "created"
                    else: raise
                if vector_db.is_ready(): vector_db.add(f"GitHub file {action}: {repo.full_name}/{fp}", {"type": "github_action", "action": "write_file", "repo": repo.full_name, "path": fp, "time": datetime.now().isoformat()})
                return f"File '{fp}' {action} in '{repo.full_name}'. Commit: {commit['commit'].sha}"
            elif operation == "list_files":
                path = kwargs.get("path", ""); br = kwargs.get("branch", repo.default_branch); logger.info(f"Listing files in '{repo.full_name}/{path}' branch '{br}'")
                contents = repo.get_contents(path, ref=br)
                if not contents: return f"Directory '{path}' empty/not found."
                file_list = [f"- {'[DIR] ' if item.type == 'dir' else ''}{item.path}" for item in contents]
                return f"Files/Dirs in '{repo.full_name}/{path}':\n" + "\n".join(file_list)
            elif operation == "clone_repo":
                # !!! WARNING: This uses subprocess and requires git command !!!
                # !!! May fail in restricted environments like default Codespaces/Colab !!!
                clone_dir = Path(tempfile.mkdtemp()); repo_url = repo.clone_url; logger.info(f"Cloning '{repo.full_name}' to {clone_dir}")
                try:
                    process = subprocess.run(['git', 'clone', repo_url, str(clone_dir)], capture_output=True, text=True, check=True, timeout=120) # Added timeout
                    logger.info(f"Clone success: {process.stdout}"); result = f"Repo '{repo.full_name}' cloned temporarily to {clone_dir} (now removed)."
                    shutil.rmtree(clone_dir); return result
                except subprocess.CalledProcessError as e: logger.error(f"Git clone failed: {e.stderr}"); shutil.rmtree(clone_dir); raise GitHubToolError(f"Git clone failed: {e.stderr}")
                except subprocess.TimeoutExpired: logger.error("Git clone timed out."); shutil.rmtree(clone_dir); raise GitHubToolError("Git clone operation timed out.")
                except FileNotFoundError: logger.error("Git command not found."); shutil.rmtree(clone_dir); raise GitHubToolError("Git command not found.")
                except Exception as e: logger.error(f"Clone/cleanup error: {e}"); shutil.rmtree(clone_dir); raise GitHubToolError(f"Cloning error: {str(e)}")
            else: raise GitHubToolError(f"Operation '{operation}' not recognized.")
        except GithubException as e: logger.error(f"GitHub API error: {e}"); raise GitHubToolError(f"GitHub API error: {e.status} - {e.data.get('message', str(e))}")
        except Exception as e: logger.error(f"GitHub tool error: {e}", exc_info=True); raise GitHubToolError(f"Unexpected GitHub tool error: {str(e)}")

class DataVisualizationTool(Tool):
    def __init__(self):
        super().__init__(
            name="visualize_data",
            description="Creates data visualizations using various plotting libraries",
            parameters={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "JSON string containing data to visualize. For arrays/lists, format as: [[x1,y1], [x2,y2], ...]"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter", "bar", "histogram", "heatmap", "box"],
                        "description": "Type of visualization to create"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot"
                    },
                    "x_label": {
                        "type": "string",
                        "description": "Label for x-axis"
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Label for y-axis"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output file path (png/jpg/pdf)"
                    }
                },
                "required": ["data", "plot_type", "output_file"]
            }
        )
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        import json
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        try:
            # Parse data
            data = json.loads(kwargs.get("data"))
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                df = pd.DataFrame(data, columns=['x', 'y'])
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                raise ToolExecutionError("Invalid data format. Expected list of [x,y] pairs or dict")
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plot_type = kwargs.get("plot_type")
            
            if plot_type == "line":
                plt.plot(df['x'], df['y'])
            elif plot_type == "scatter":
                plt.scatter(df['x'], df['y'])
            elif plot_type == "bar":
                plt.bar(df['x'], df['y'])
            elif plot_type == "histogram":
                plt.hist(df['y'], bins=30)
            elif plot_type == "heatmap":
                pivot = df.pivot(index='x', columns='y', values='value')
                sns.heatmap(pivot, annot=True)
            elif plot_type == "box":
                sns.boxplot(x='x', y='y', data=df)
            
            # Customize plot
            plt.title(kwargs.get("title", ""))
            plt.xlabel(kwargs.get("x_label", "X"))
            plt.ylabel(kwargs.get("y_label", "Y"))
            plt.tight_layout()
            
            # Save plot
            output_file = kwargs.get("output_file")
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Plot saved to: {output_file}")
            return f"Visualization created and saved to {output_file}"
            
        except json.JSONDecodeError as e:
            logger.error(f"Data parsing error: {e}")
            raise ToolExecutionError(f"Invalid JSON data format: {e}")
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
            raise ToolExecutionError(f"Failed to create visualization: {e}")

class AWSRekognitionTool(Tool):
    def __init__(self):
        super().__init__(
            name="aws_rekognition",
            description="Perform facial recognition and image analysis using AWS Rekognition",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["detect_faces", "compare_faces", "detect_labels", "detect_text"],
                        "description": "Type of Rekognition operation to perform"
                    },
                    "source_image": {
                        "type": "string",
                        "description": "Path to source image file or base64-encoded image"
                    },
                    "target_image": {
                        "type": "string",
                        "description": "Path to target image for face comparison (only for compare_faces)"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold for face comparison (0-100)"
                    }
                },
                "required": ["operation", "source_image"]
            }
        )
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        import boto3
        from pathlib import Path
        import base64
        
        try:
            # Initialize Rekognition client
            rekognition = boto3.client('rekognition')
            operation = kwargs.get("operation")
            
            # Process source image
            source_image = kwargs.get("source_image")
            source_bytes = None
            
            if source_image.startswith('data:image'):
                # Handle base64 encoded image
                _, encoded = source_image.split(',', 1)
                source_bytes = base64.b64decode(encoded)
            else:
                # Handle file path
                with open(source_image, 'rb') as f:
                    source_bytes = f.read()
            
            if operation == "detect_faces":
                response = rekognition.detect_faces(
                    Image={'Bytes': source_bytes},
                    Attributes=['ALL']
                )
                return self._format_face_detection(response)
                
            elif operation == "compare_faces":
                if not kwargs.get("target_image"):
                    raise ToolExecutionError("target_image required for face comparison")
                    
                target_image = kwargs.get("target_image")
                target_bytes = None
                
                if target_image.startswith('data:image'):
                    _, encoded = target_image.split(',', 1)
                    target_bytes = base64.b64decode(encoded)
                else:
                    with open(target_image, 'rb') as f:
                        target_bytes = f.read()
                
                response = rekognition.compare_faces(
                    SourceImage={'Bytes': source_bytes},
                    TargetImage={'Bytes': target_bytes},
                    SimilarityThreshold=kwargs.get('similarity_threshold', 80)
                )
                return self._format_face_comparison(response)
                
            elif operation == "detect_labels":
                response = rekognition.detect_labels(
                    Image={'Bytes': source_bytes},
                    MaxLabels=10
                )
                return self._format_label_detection(response)
                
            elif operation == "detect_text":
                response = rekognition.detect_text(
                    Image={'Bytes': source_bytes}
                )
                return self._format_text_detection(response)
                
        except boto3.exceptions.BotoServerError as e:
            logger.error(f"AWS Rekognition API error: {e}")
            raise ToolExecutionError(f"AWS Rekognition API error: {e}")
        except Exception as e:
            logger.error(f"Image processing error: {e}", exc_info=True)
            raise ToolExecutionError(f"Failed to process image: {e}")
    
    def _format_face_detection(self, response):
        faces = response.get('FaceDetails', [])
        results = []
        for face in faces:
            results.append({
                'confidence': face.get('Confidence'),
                'age_range': face.get('AgeRange'),
                'gender': face.get('Gender', {}).get('Value'),
                'emotions': [e.get('Type') for e in face.get('Emotions', [])],
                'pose': face.get('Pose')
            })
        return f"Detected {len(faces)} faces:\n" + json.dumps(results, indent=2)
    
    def _format_face_comparison(self, response):
        matches = response.get('FaceMatches', [])
        return f"Found {len(matches)} matching faces with similarities: " + \
               ', '.join([f"{match['Similarity']:.1f}%" for match in matches])
    
    def _format_label_detection(self, response):
        labels = response.get('Labels', [])
        return "Detected labels:\n" + \
               '\n'.join([f"- {l['Name']} ({l['Confidence']:.1f}%)" for l in labels])
    
    def _format_text_detection(self, response):
        texts = response.get('TextDetections', [])
        return "Detected text:\n" + \
               '\n'.join([f"- {t['DetectedText']} ({t['Confidence']:.1f}%)" for t in texts])

class ImageGenerationTool(Tool):
    def __init__(self):
        super().__init__(
            name="generate_image",
            description="Generate images using Stability AI's image generation API",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Name for the output file (without extension)"
                    }
                },
                "required": ["prompt", "file_name"]
            }
        )
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        import requests
        from pathlib import Path
        import os
        
        try:
            prompt = kwargs.get("prompt")
            file_name = kwargs.get("file_name")
            output_dir = "generated_images"  # Fixed output directory
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            if not stability_api_key:
                raise ToolExecutionError("Stability AI API key missing")
            
            logger.info(f"Generating image for prompt: {prompt}")
            
            # Generate image using Stability AI API
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/core",
                headers={
                    "authorization": f"Bearer {stability_api_key}",
                    "accept": "image/*"
                },
                files={"none": ""},
                data={
                    "prompt": prompt,
                    "output_format": "jpeg"
                }
            )
            
            if response.status_code != 200:
                raise ToolExecutionError(f"API Error: {response.json()}")
            
            # Save the image
            output_path = Path(output_dir) / f"{file_name}.jpeg"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Image saved to: {output_path}")
            
            # Store in vector DB if available
            if vector_db.is_ready():
                vector_db.add(
                    f"Generated image from prompt: {prompt}",
                    {
                        "type": "generated_image",
                        "prompt": prompt,
                        "file_path": str(output_path),
                        "time": datetime.now().isoformat()
                    }
                )
            
            return f"Image generated and saved to {output_path}"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            raise ToolExecutionError(f"Failed to connect to Stability AI API: {e}")
        except Exception as e:
            logger.error(f"Image generation error: {e}", exc_info=True)
            raise ToolExecutionError(f"Failed to generate image: {e}")

# --- Initialize Tools ---
def initialize_tools():
    logger.info("Initializing tools...")
    tools_list = [
        WeatherTool(),
        SearchTool(),
        WebScraperTool(),
        CodeExecutionTool(),
        DateTimeTool(),
        GitHubTool(),
        DataVisualizationTool(),
        AWSRekognitionTool(),
        ImageGenerationTool()
    ]
    if vector_db.is_ready():
        tools_list.append(VectorSearchTool())
        logger.info("Vector search tool initialized.")
    else:
        logger.warning("Vector search tool NOT initialized.")
    schemas = [t.get_schema() for t in tools_list]
    tool_map = {t.name: t for t in tools_list}
    logger.info(f"Tools initialized: {list(tool_map.keys())}")
    return schemas, tool_map
active_tool_schemas, tool_map = initialize_tools()

# --- System Message Definition ---
SYSTEM_MESSAGE = { 
    "role": "system", 
    "content": (
        "You are OmniBot. Use tools proactively. **Never** say you lack access; state you will use a tool. "
        "Use `code_execution` to acknowledge Python code requests (execution handled externally). "
        "Use `github_operations` for GitHub tasks. "
        "Use `semantic_memory_search` for past info if available.\n\n"
        "**File Handling Capabilities:**\n"
        "- You can analyze images, documents, and other files provided via URL or direct upload\n"
        "- When a user shares a file, carefully examine its contents and provide insights\n"
        "- For images, describe what you see in detail\n"
        "- For documents, summarize and analyze the content\n\n"
        "**Available Tools:**\n" + 
        "".join([f"- `{tool.name}`: {tool.description}\n" for tool in tool_map.values()]) + 
        "\nAnswer based on tool outputs and file analysis."
        "You work as a part time LinkedIn and X (Twitter) content creator for a tech company, they are called Raiden Agents.. they build AI agents for business and companies to automate their workflows.\n"

    )
}
logger.info(f"System message generated. Approx tokens: {len(SYSTEM_MESSAGE['content']) // 4}")

# --- File Processing Helper (Updated) ---
def process_file_input(file_identifier):
    """Process file input from URL, GCS, or local path - enhanced version with better error handling."""
    logger.info(f"Processing file: {file_identifier}")
    content_part = {"type": "file"}
    file_data_dict = {}
    mime_type = None
    
    # URL-based file handling
    if file_identifier.startswith("http://") or file_identifier.startswith("https://"):
        file_data_dict["file_id"] = file_identifier
        mime_type, _ = mimetypes.guess_type(file_identifier)
        if not mime_type:
            try:
                r = requests.head(file_identifier, allow_redirects=True, timeout=10)
                r.raise_for_status()
                ct = r.headers.get('Content-Type')
                mime_type = ct.split(';')[0].strip() if ct else None
                # If HEAD request worked but didn't get mime type, try with GET to fetch content
                if not mime_type:
                    r = requests.get(file_identifier, timeout=15)
                    r.raise_for_status()
                    ct = r.headers.get('Content-Type')
                    mime_type = ct.split(';')[0].strip() if ct else "application/octet-stream"
                    # Try to download the content directly
                    file_content = r.content
                    encoded_content = base64.b64encode(file_content).decode("utf-8")
                    file_data_dict["file_data"] = f"data:{mime_type};base64,{encoded_content}"
                    logger.info(f" Downloaded URL content: {len(file_content)} bytes with mime: {mime_type}")
                    # Skip file_id approach and use direct content
                    del file_data_dict["file_id"]
            except requests.exceptions.RequestException as e:
                logger.warning(f" URL request failed: {e}")
                # Still try to proceed with best effort
        
        if "file_id" in file_data_dict and mime_type:
            file_data_dict["format"] = mime_type
            logger.info(f" URL MIME: {mime_type}")
        elif "file_id" in file_data_dict:
            # Default mime type if we couldn't detect it
            mime_type = "application/octet-stream"
            file_data_dict["format"] = mime_type
            logger.warning(f" Default MIME for URL: {mime_type}")
    
    # Google Cloud Storage handling
    elif file_identifier.startswith("gs://"):
        file_data_dict["file_id"] = file_identifier
        mime_type, _ = mimetypes.guess_type(file_identifier)
        if mime_type:
            file_data_dict["format"] = mime_type
            logger.info(f" GCS MIME: {mime_type}")
        else:
            # Default mime type if we couldn't detect it
            mime_type = "application/octet-stream"
            file_data_dict["format"] = mime_type
            logger.warning(f" Default MIME for GCS: {mime_type}")
    
    # Local file handling
    else:
        lp = Path(file_identifier)
        if not lp.is_file():
            logger.error(f"Local file not found: {lp}")
            return None
        
        try:
            fb = lp.read_bytes()
            ed = base64.b64encode(fb).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(lp)
            if not mime_type:
                # Try to guess from content
                import magic  # This might require installation
                try:
                    mime_type = magic.from_buffer(fb, mime=True)
                except (ImportError, AttributeError):
                    # Fallback to some basic checks
                    if fb.startswith(b'\xff\xd8\xff'):
                        mime_type = "image/jpeg"
                    elif fb.startswith(b'\x89PNG\r\n\x1a\n'):
                        mime_type = "image/png"
                    elif fb.startswith(b'GIF87a') or fb.startswith(b'GIF89a'):
                        mime_type = "image/gif"
                    elif fb.startswith(b'%PDF-'):
                        mime_type = "application/pdf"
                    else:
                        mime_type = "application/octet-stream"
                logger.warning(f" Guessed MIME: {mime_type}")
            else:
                logger.info(f" Local MIME: {mime_type}")
            
            file_data_dict["file_data"] = f"data:{mime_type};base64,{ed}"
            if vector_db.is_ready():
                fn = lp.name
                vector_db.add(f"User file: {fn} ({mime_type})", {
                    "type": "file_provided", 
                    "source": "local", 
                    "filename": fn, 
                    "mime_type": mime_type, 
                    "time": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Read local file error {lp}: {e}")
            traceback.print_exc()
            return None
    
    content_part["file"] = file_data_dict
    return content_part

# --- Tool Execution Wrapper (Accepts Dict) ---
def execute_tool_call(tool_call_data):
    """Wrapper for executing tool calls using dictionary input."""
    function_name = tool_call_data.get('function', {}).get('name')
    arguments_str = tool_call_data.get('function', {}).get('arguments')
    if not function_name: error_msg = "Error: Tool call missing function name."; logger.error(error_msg); return error_msg
    try:
        function_args = json.loads(arguments_str) if arguments_str else {}
        logger.info(f"Attempting execution: '{function_name}' args: {function_args}") # File only
    except json.JSONDecodeError: error_msg = f"Error: Invalid JSON args for {function_name}"; logger.error(error_msg); return error_msg # Console ERROR
    if function_name not in tool_map: error_msg = f"Error: Unknown function '{function_name}'"; logger.error(error_msg); return error_msg # Console ERROR
    try:
        tool = tool_map[function_name]; result = tool.execute(**function_args)
        logger.info(f"Tool '{function_name}' executed successfully.") # File only
        logger.debug(f"Tool '{function_name}' result snippet: {str(result)[:200]}...") # File only
        return result
    except ToolExecutionError as e: logger.error(f"Tool execution failed '{function_name}': {e}"); return f"Error executing tool {function_name}: {e}" # Console ERROR + return error msg
    except Exception as e: logger.critical(f"Unexpected critical error executing tool '{function_name}'", exc_info=True); return f"Critical Error executing tool {function_name}." # Console CRITICAL + return error msg


# --- Handle Streaming Response ---
# (Unchanged)
def handle_streaming_response(stream):
    full_response_content = ""; tool_calls_agg = defaultdict(lambda: {"id": None, "name": None, "arguments": ""}); final_tool_calls_list = []; completed_tool_call_indices = set(); current_tool_call_index = -1
    print("\nOmniBot: ", end="", flush=True) # Console output
    try:
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content: print(delta_content, end="", flush=True); full_response_content += delta_content
            delta_tool_calls = chunk.choices[0].delta.tool_calls
            if delta_tool_calls:
                for tc_chunk in delta_tool_calls:
                    idx = tc_chunk.index
                    if tc_chunk.id: tool_calls_agg[idx]["id"] = tc_chunk.id
                    if tc_chunk.function and tc_chunk.function.name: tool_calls_agg[idx]["name"] = tc_chunk.function.name
                    if tc_chunk.function and tc_chunk.function.arguments: tool_calls_agg[idx]["arguments"] += tc_chunk.function.arguments
                    current_call = tool_calls_agg[idx]
                    if current_call["id"] and current_call["name"] and idx not in completed_tool_call_indices:
                         args_str = current_call["arguments"]; is_complete_json = False
                         try: json.loads(args_str); is_complete_json = True
                         except json.JSONDecodeError: pass
                         if is_complete_json:
                              logger.debug(f"Stream: Finalizing tool {idx}...") # File only
                              final_tool_calls_list.append({"id": current_call["id"], "type": "function", "function": {"name": current_call["name"], "arguments": args_str}})
                              completed_tool_call_indices.add(idx)
    except Exception as e: logger.error(f"Stream error: {e}", exc_info=True); print(f"\n[Stream Error: {e}]") # Console ERROR
    finally: print()
    final_message_dict = {"role": "assistant", "content": full_response_content if full_response_content else None, "tool_calls": final_tool_calls_list if final_tool_calls_list else None}
    return final_message_dict


# --- Main Chat Loop ---
def chat_agent():
    model_name = "gemini/gemini-2.0-flash"
    memory = ConversationMemory(system_message=SYSTEM_MESSAGE, max_tokens=1_000_000) # Increased limit

    logger.info("\n--- OmniBot Initialized (v9.8 - Codespaces Ready) ---") # File only
    print(f"OmniBot v9.8 Initialized. Model: {model_name}. Type 'quit' to exit.") # Console output
    print(f"Vector DB Status: {'Ready' if vector_db.is_ready() else 'Unavailable'}") # Console output
    print("-" * 60 + "\n") # Console output

    while True:
        try:
            user_input = input("You: ") # Console output
            if user_input.lower() == "quit": logger.info("User quit."); break # File only

            user_message_content = []
            # Enhanced file handling
            if user_input.lower().startswith("file:") and len(user_input.split(' ', 1)) > 1:
                file_id = user_input.split(' ', 1)[1].strip()
                if file_id:
                    print("OmniBot: Processing file, please wait...", flush=True)  # Console output
                    file_part = process_file_input(file_id)
                    if not file_part:
                        print("OmniBot: [Error processing file. Please check if the file exists or URL is accessible.]")
                        continue  # Console output
                    
                    prompt = input("You (prompt for file): ")  # Console output
                    if prompt:
                        user_message_content.extend([{"type": "text", "text": prompt}, file_part])
                    else:
                        # Allow file without prompt - default prompt
                        user_message_content.extend([{"type": "text", "text": "Analyze this file for me."}, file_part])
                else:
                    print("OmniBot: [File command needs path/URL.]")
                    continue  # Console output
            else:
                user_message_content.append({"type": "text", "text": user_input})
            
            if not user_message_content:
                continue

            user_message = {"role": "user", "content": user_message_content}
            if not memory.add_message(user_message): print("OmniBot: [Message too long.]"); continue # Console output

            user_text = user_input
            if isinstance(user_message_content, list):
                for item in user_message_content:
                    if item.get("type") == "text": user_text = item.get("text",""); break
            if vector_db.is_ready(): vector_db.add(f"User said: {user_text}", {"type": "user_message", "time": datetime.now().isoformat()})

            # --- CONSOLE OUTPUT: Thinking ---
            print("\nOmniBot: Thinking...", flush=True)
            logger.info("OmniBot: Thinking...") # File only

            current_messages = memory.get_messages()
            response_stream = litellm.completion(model=model_name, messages=current_messages, tools=active_tool_schemas, tool_choice="auto", stream=True)

            # Prints stream to console via handle_streaming_response
            response_message_dict = handle_streaming_response(response_stream)
            memory.add_message(response_message_dict)

            if response_message_dict.get("tool_calls"):
                # --- CONSOLE OUTPUT: Tool Usage ---
                print(f"OmniBot: Using {len(response_message_dict['tool_calls'])} tool(s)...", flush=True)
                logger.info(f"LLM requested {len(response_message_dict['tool_calls'])} tool(s)...") # File only

                tool_results = []
                for tc_data in response_message_dict["tool_calls"]:
                    # Pass dictionary directly to corrected execute_tool_call
                    result_content = execute_tool_call(tc_data)
                    if isinstance(result_content, str) and result_content.lower().startswith("error"):
                        logger.warning(f"Tool '{tc_data.get('function', {}).get('name')}' failed. Error: {result_content}") # Console WARN + File
                    result_msg = {"role": "tool", "tool_call_id": tc_data.get('id'), "content": str(result_content)}
                    tool_results.append(result_msg)
                    memory.add_message(result_msg) # Add result to memory

                # --- CONSOLE OUTPUT: Processing ---
                print("\nOmniBot: Processing tool results...", flush=True)
                logger.info("OmniBot: Processing tool results...") # File only

                messages_with_results = memory.get_messages()
                final_stream = litellm.completion(model=model_name, messages=messages_with_results, stream=True)

                # Prints stream to console via handle_streaming_response
                final_response_dict = handle_streaming_response(final_stream)
                memory.add_message(final_response_dict)

                final_content = final_response_dict.get("content", "")
                # Logging of final content happens below
                if final_content and vector_db.is_ready(): vector_db.add(f"OmniBot response: {final_content}", {"type": "assistant_response", "after_tool_use": True, "time": datetime.now().isoformat()})
                if final_content: logger.info(f"OmniBot final response (after tools): {final_content}") # File only
                elif response_message_dict.get("tool_calls"): logger.info("OmniBot final response after tool use had no text.") # File only
                else: logger.warning("OmniBot final response was empty.") # Console WARN

            elif response_message_dict.get("content"):
                assistant_content = response_message_dict["content"]
                logger.info(f"OmniBot response: {assistant_content}") # File only
                if vector_db.is_ready(): vector_db.add(f"OmniBot response: {assistant_content}", {"type": "assistant_response", "after_tool_use": False, "time": datetime.now().isoformat()})
            else:
                logger.warning("OmniBot received empty initial response (no content or tools).") # Console WARN

        # --- Error Handling ---
        except litellm.exceptions.APIError as e:
             logger.error(f"LiteLLM API Error: {e}", exc_info=True); print(f"\n!!! OmniBot Error: API Failure {e.status_code if hasattr(e, 'status_code') else ''} !!!", file=sys.stderr) # Console ERROR
             try: logger.error(f"API Error Body: {json.dumps(e.response.json(), indent=2)}") # Console ERROR
             except: logger.error(f"Raw API Error: {e.response.text if hasattr(e, 'response') else 'N/A'}") # Console ERROR
        except AgentException as e: logger.error(f"Agent Error: {e}", exc_info=True); print(f"\n!!! OmniBot Error: {e} !!!", file=sys.stderr) # Console ERROR
        except KeyboardInterrupt: logger.info("User interrupted."); print("\nOmniBot: Exiting..."); break # File info, Console output
        except Exception as e: logger.critical(f"Critical error in main loop!", exc_info=True); print(f"\n!!! OmniBot Critical Error: {e} !!!", file=sys.stderr); break # Console CRITICAL


# --- Start the Agent ---
if __name__ == "__main__":
    try:
        chat_agent()
    except APIKeyError:
        print("Execution stopped: Missing critical API key. Please set it in .env", file=sys.stderr) # Console output
    except Exception as main_e:
        logger.critical(f"Critical agent startup error: {main_e}", exc_info=True) # File CRITICAL
        print(f"\n!!! Critical startup error: {main_e}. Check logs for details. !!!", file=sys.stderr) # Console CRITICAL