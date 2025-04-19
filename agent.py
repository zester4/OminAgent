# agent.py
# -*- coding: utf-8 -*-
"""
OmniBot Agent Logic v9.8 (Refactored for API use)

Core Features:
- Model: gemini/gemini-2.0-flash
- Tools: Weather, Search, Firecrawl, GitHub, CodeExec(Ack), DateTime, VectorSearch
- Vector DB (in-memory), Conversation Memory Pruning.
- Structured Logging (Detailed logs -> file).
- File Handling capability (input path/URL).
"""

# Standard library imports first
import base64
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

# Third-party imports
import litellm
import numpy as np
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from firecrawl import FirecrawlApp
from github import Github, GithubException
from sentence_transformers import SentenceTransformer # Import moved here

# --- Setup Logging (Clean Console) ---
def setup_logging():
    # Configuration for logging agent activity primarily to a file
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("gemini_agent_logic") # Specific name
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("gemini_agent_logic.log") # Log file name
    file_handler.setFormatter(log_formatter); file_handler.setLevel(logging.INFO)
    # Only add file handler by default for the agent logic part
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
logger = setup_logging()
logger.info("--- Loading Gemini Agent Logic ---")

# --- Custom Exceptions ---
class AgentException(Exception): pass
class ToolExecutionError(AgentException): pass
class APIKeyError(AgentException): pass
class VectorDBError(AgentException): pass
class GitHubToolError(ToolExecutionError): pass

# --- Load Environment Variables ---
load_dotenv()
logger.info("Attempted to load environment variables from .env file.")

# --- API Key Setup (from Environment) ---
logger.info("Setting up API Keys from environment...")
def get_required_key(env_var):
    key = os.environ.get(env_var)
    if not key: logger.error(f"CRITICAL: {env_var} not set."); raise APIKeyError(f"Missing required API key: {env_var}")
    logger.info(f"Found required key: {env_var}")
    return key
def get_optional_key(env_var):
    key = os.environ.get(env_var)
    if not key: logger.warning(f"Optional key {env_var} not set.")
    else: logger.info(f"Found optional key: {env_var}")
    return key
try:
    gemini_api_key = get_required_key("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = gemini_api_key # Ensure LiteLLM sees it
    openweathermap_api_key = get_optional_key("OPENWEATHERMAP_API_KEY")
    firecrawl_api_key = get_optional_key("FIRECRAWL_API_KEY")
    github_api_key = get_optional_key("GITHUB_API_KEY")
except APIKeyError as e: logger.critical(f"API Key Error: {e}"); raise
except Exception as e: logger.critical(f"API key setup error: {e}", exc_info=True); raise
logger.info("API Keys configured.")


# --- Memory Management ---
class ConversationMemory:
    CHARS_PER_TOKEN_ESTIMATE = 4
    def __init__(self, max_tokens=1_000_000, system_message=None):
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
    def get_messages(self): return list(self.messages) # Return a copy
    def get_last_user_message_content(self):
        for msg in reversed(self.messages):
            if msg.get("role") == "user": return msg.get("content")
        return None

# --- Vector Database for Semantic Memory ---
class VectorDB:
    def __init__(self):
        self.model = None; self.texts = []; self.metadata = []; self.embeddings = None; self.initialized = False
        try: self.model = SentenceTransformer('all-MiniLM-L6-v2'); self.initialized = True; logger.info("Vector DB init success")
        except ImportError: logger.error("sentence-transformers not installed. VDB disabled.")
        except Exception as e: logger.error(f"SentenceTransformer load failed: {e}. VDB disabled.")
    def add(self, text, metadata=None):
        if not self.is_ready(): return False
        if not text or not isinstance(text, str): return False
        try:
            embedding = self.model.encode(text)
            if self.embeddings is None: self.embeddings = np.array([embedding])
            else:
                if not isinstance(self.embeddings, np.ndarray): logger.error("VDB state error."); return False
                self.embeddings = np.vstack([self.embeddings, embedding])
            self.texts.append(text); self.metadata.append(metadata or {})
            logger.debug(f"Added VDB entry: {text[:50]}...")
            return True
        except Exception as e: logger.error(f"VDB add error: {e}", exc_info=True); return False
    def search(self, query, top_k=3):
        if not self.is_ready(): raise VectorDBError("VDB not initialized")
        if self.embeddings is None or self.embeddings.size == 0: return []
        try:
            q_emb = self.model.encode(query); embs_2d = self.embeddings if self.embeddings.ndim == 2 else self.embeddings.reshape(-1, self.model.get_sentence_embedding_dimension())
            norms_e = np.linalg.norm(embs_2d, axis=1); norm_q = np.linalg.norm(q_emb)
            if norm_q == 0: return []
            valid_idx = np.where(norms_e > 1e-8)[0]
            if len(valid_idx) == 0: return []
            valid_embs = embs_2d[valid_idx]; valid_norms = norms_e[valid_idx]
            sims = np.dot(valid_embs, q_emb) / (valid_norms * norm_q)
            count = min(top_k, len(sims));
            if count <= 0: return []
            rel_top_idx = np.argsort(sims)[::-1][:count]; orig_top_idx = valid_idx[rel_top_idx]
            results = []
            for i in orig_top_idx:
                 if i < len(self.texts): results.append({ "text": self.texts[i], "similarity": float(sims[rel_top_idx[np.where(orig_top_idx == i)[0][0]]]), "metadata": self.metadata[i] })
            logger.info(f"VDB search '{query[:30]}...' returned {len(results)}.")
            return results
        except Exception as e: logger.error(f"VDB search error: {e}", exc_info=True); raise VectorDBError(f"VDB search failed: {e}")
    def is_ready(self): return self.initialized and self.model is not None
# Global VDB instance for the agent's lifetime
vector_db_instance = VectorDB()

# --- Tool Implementation (Object-Oriented) ---
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
                if vector_db_instance.is_ready(): vector_db_instance.add(f"Weather: {l}({u}): {res}", {"type": "weather", "location": l, "time": datetime.now().isoformat()})
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
                if vector_db_instance.is_ready(): vector_db_instance.add(f"Search snippet '{q}': {r.get('title', '')} - {r.get('body', '')}", {"type": "search_result", "url": r.get('href'), "query": q, "time": datetime.now().isoformat()})
            return f"Search results for '{q}':\n\n" + "\n\n---\n\n".join(fmt)
        except Exception as e: logger.error(f"Search error: {e}"); raise ToolExecutionError(f"Search failed: {e}")
class WebScraperTool(Tool):
    def __init__(self): super().__init__(name="scrape_website_for_llm", description="Fetches main content of a specific URL as Markdown.", parameters={"type": "object", "properties": {"url": {"type": "string", "description": "URL to scrape."}}}, required=["url"])
    def execute(self, **kwargs):
        self.validate_args(kwargs); url = kwargs.get("url"); logger.info(f"Scraping URL: {url}")
        if not firecrawl_api_key: raise ToolExecutionError("Firecrawl API key missing.")
        try:
            app = FirecrawlApp(api_key=firecrawl_api_key)
            scraped_data = app.scrape_url(url=url, params={'formats': ['markdown']})
            markdown_content = None
            if isinstance(scraped_data, dict): markdown_content = scraped_data.get('markdown')
            elif isinstance(scraped_data, str): markdown_content = scraped_data
            if markdown_content:
                logger.info(f"Scrape success: {url}")
                if vector_db_instance.is_ready():
                    chunks = self._chunk_content(markdown_content); logger.info(f"Storing {len(chunks)} chunks from {url} in VDB.")
                    for i, chunk in enumerate(chunks): vector_db_instance.add(chunk, {"type": "web_content", "url": url, "chunk": i+1, "total_chunks": len(chunks), "time": datetime.now().isoformat()})
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
        if not vector_db_instance or not vector_db_instance.is_ready(): raise ToolExecutionError("Vector DB unavailable.")
        try:
            res = vector_db_instance.search(q, top_k=c)
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
                if vector_db_instance.is_ready(): vector_db_instance.add(f"Created GitHub repo: {repo.full_name}", {"type": "github_action", "action": "create_repo", "repo": repo.full_name, "time": datetime.now().isoformat()})
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
                if vector_db_instance.is_ready(): vector_db_instance.add(f"GitHub file {action}: {repo.full_name}/{fp}", {"type": "github_action", "action": "write_file", "repo": repo.full_name, "path": fp, "time": datetime.now().isoformat()})
                return f"File '{fp}' {action} in '{repo.full_name}'. Commit: {commit['commit'].sha}"
            elif operation == "list_files":
                path = kwargs.get("path", ""); br = kwargs.get("branch", repo.default_branch); logger.info(f"Listing files in '{repo.full_name}/{path}' branch '{br}'")
                contents = repo.get_contents(path, ref=br)
                if not contents: return f"Directory '{path}' empty/not found."
                file_list = [f"- {'[DIR] ' if item.type == 'dir' else ''}{item.path}" for item in contents]
                return f"Files/Dirs in '{repo.full_name}/{path}':\n" + "\n".join(file_list)
            elif operation == "clone_repo":
                clone_dir = Path(tempfile.mkdtemp()); repo_url = repo.clone_url; logger.info(f"Cloning '{repo.full_name}' to {clone_dir}")
                try:
                    process = subprocess.run(['git', 'clone', repo_url, str(clone_dir)], capture_output=True, text=True, check=True, timeout=120)
                    logger.info(f"Clone success: {process.stdout}"); result = f"Repo '{repo.full_name}' cloned temporarily to {clone_dir} (now removed)."
                    shutil.rmtree(clone_dir); return result
                except subprocess.CalledProcessError as e: logger.error(f"Git clone failed: {e.stderr}"); shutil.rmtree(clone_dir); raise GitHubToolError(f"Git clone failed: {e.stderr}")
                except subprocess.TimeoutExpired: logger.error("Git clone timed out."); shutil.rmtree(clone_dir); raise GitHubToolError("Git clone operation timed out.")
                except FileNotFoundError: logger.error("Git command not found."); shutil.rmtree(clone_dir); raise GitHubToolError("Git command not found.")
                except Exception as e: logger.error(f"Clone/cleanup error: {e}"); shutil.rmtree(clone_dir); raise GitHubToolError(f"Cloning error: {str(e)}")
            else: raise GitHubToolError(f"Operation '{operation}' not recognized.")
        except GithubException as e: logger.error(f"GitHub API error: {e}"); raise GitHubToolError(f"GitHub API error: {e.status} - {e.data.get('message', str(e))}")
        except Exception as e: logger.error(f"GitHub tool error: {e}", exc_info=True); raise GitHubToolError(f"Unexpected GitHub tool error: {str(e)}")


# --- Initialize Tools ---
def initialize_tools():
    logger.info("Initializing tools..."); tools_list = [ WeatherTool(), SearchTool(), WebScraperTool(), CodeExecutionTool(), DateTimeTool(), GitHubTool() ]
    if vector_db_instance.is_ready(): tools_list.append(VectorSearchTool()); logger.info("Vector search tool initialized.")
    else: logger.warning("Vector search tool NOT initialized.")
    schemas = [t.get_schema() for t in tools_list]; tool_map = {t.name: t for t in tools_list}
    logger.info(f"Tools initialized: {list(tool_map.keys())}")
    return schemas, tool_map
# Global tool map and schemas for the agent instance
active_tool_schemas_global, tool_map_global = initialize_tools()

# --- System Message Definition ---
SYSTEM_MESSAGE = { "role": "system", "content": ("You are OmniBot. Use tools proactively. **Never** say you lack access; state you will use a tool. Use `code_execution` to acknowledge Python code requests (execution handled externally). Use `github_operations` for GitHub tasks. Use `semantic_memory_search` for past info if available.\n\n**Available Tools:**\n" + "".join([f"- `{tool.name}`: {tool.description}\n" for tool in tool_map_global.values()]) + "\nAnswer based on tool outputs.")}
logger.info(f"System message generated. Approx tokens: {len(SYSTEM_MESSAGE['content']) // 4}")


# --- File Processing Helper ---
def process_file_input(file_identifier):
    # NOTE: This function currently works with local paths relative to where the agent.py script is running.
    # In a web server context, handling file uploads requires a different approach.
    # This implementation assumes the path is accessible from the server process.
    logger.info(f"Processing file identifier (path/URL): {file_identifier}");
    content_part = {"type": "file"}; file_data_dict = {}; mime_type = None
    if file_identifier.startswith("http://") or file_identifier.startswith("https://"):
        file_data_dict["file_id"] = file_identifier; mime_type, _ = mimetypes.guess_type(file_identifier)
        if not mime_type:
            try: r=requests.head(file_identifier, allow_redirects=True, timeout=7); r.raise_for_status(); ct=r.headers.get('Content-Type'); mime_type=ct.split(';')[0].strip() if ct else None
            except requests.exceptions.RequestException as e: logger.warning(f" HEAD failed: {e}")
        if mime_type: file_data_dict["format"] = mime_type; logger.info(f" URL MIME: {mime_type}")
        else: logger.warning(" No MIME for URL.")
    elif file_identifier.startswith("gs://"): # Google Cloud Storage handling (ensure auth)
        file_data_dict["file_id"] = file_identifier; mime_type, _ = mimetypes.guess_type(file_identifier)
        if mime_type: file_data_dict["format"] = mime_type; logger.info(f" GCS MIME: {mime_type}")
    else: # Assume local file path relative to server
        lp = Path(file_identifier)
        # Security: Add checks here to prevent accessing files outside allowed directories if needed
        if not lp.is_file(): logger.error(f"Local file not found/accessible: {lp}"); return None
        try:
            fb = lp.read_bytes(); ed = base64.b64encode(fb).decode("utf-8"); mime_type, _ = mimetypes.guess_type(lp)
            if not mime_type: mime_type = "application/octet-stream"; logger.warning(f" Default MIME: {mime_type}")
            else: logger.info(f" Local MIME: {mime_type}")
            file_data_dict["file_data"] = f"data:{mime_type};base64,{ed}";
            if vector_db_instance.is_ready(): fn = lp.name; vector_db_instance.add(f"Processed local file: {fn} ({mime_type})", {"type": "file_processed", "source": "local", "filename": fn, "mime_type": mime_type, "time": datetime.now().isoformat()})
        except Exception as e: logger.error(f"Read local file error {lp}: {e}"); traceback.print_exc(); return None
    content_part["file"] = file_data_dict; return content_part

# --- Tool Execution Wrapper (Accepts Dict) ---
def execute_tool_call(tool_call_data):
    """Wrapper for executing tool calls using dictionary input."""
    function_name = tool_call_data.get('function', {}).get('name')
    arguments_str = tool_call_data.get('function', {}).get('arguments')
    if not function_name: error_msg = "Error: Tool call missing function name."; logger.error(error_msg); return error_msg
    try: function_args = json.loads(arguments_str) if arguments_str else {}; logger.info(f"Executing tool: '{function_name}' args: {function_args}")
    except json.JSONDecodeError: error_msg = f"Error: Invalid JSON args for {function_name}"; logger.error(error_msg); return error_msg
    if function_name not in tool_map_global: error_msg = f"Error: Unknown function '{function_name}'"; logger.error(error_msg); return error_msg
    try:
        tool = tool_map_global[function_name]; result = tool.execute(**function_args)
        logger.info(f"Tool '{function_name}' executed successfully.")
        logger.debug(f"Tool '{function_name}' result snippet: {str(result)[:200]}...")
        return result
    except ToolExecutionError as e: logger.error(f"Tool execution failed '{function_name}': {e}"); return f"Error executing tool {function_name}: {e}"
    except Exception as e: logger.critical(f"Unexpected critical error executing tool '{function_name}'", exc_info=True); return f"Critical Error executing tool {function_name}."

# --- Handle Streaming Response ---
def handle_streaming_response_dict(stream):
    """Processes LiteLLM stream and aggregates the full response dictionary."""
    full_response_content = ""; tool_calls_agg = defaultdict(lambda: {"id": None, "name": None, "arguments": ""}); final_tool_calls_list = []; completed_tool_call_indices = set(); current_tool_call_index = -1
    try:
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content: full_response_content += delta_content
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
                              logger.debug(f"Stream: Finalizing tool {idx}...")
                              final_tool_calls_list.append({"id": current_call["id"], "type": "function", "function": {"name": current_call["name"], "arguments": args_str}})
                              completed_tool_call_indices.add(idx)
    except Exception as e: logger.error(f"Stream processing error: {e}", exc_info=True)
    # Construct final message dict
    final_message_dict = {"role": "assistant", "content": full_response_content if full_response_content else None, "tool_calls": final_tool_calls_list if final_tool_calls_list else None}
    return final_message_dict

# --- Agent Class ---
class OmniBotAgent:
    def __init__(self):
        self.model_name = "gemini/gemini-2.0-flash"
        self.memory = ConversationMemory(system_message=SYSTEM_MESSAGE, max_tokens=1_000_000)
        # Tools are initialized globally, agent just uses them
        self.active_tool_schemas = active_tool_schemas_global
        self.tool_map = tool_map_global
        # VDB is also global for simplicity in this single-file structure
        self.vector_db = vector_db_instance
        logger.info(f"OmniBotAgent initialized with model {self.model_name}")

    async def process_message_stream(self, user_input, file_identifier=None):
        """Processes user message, handles files, calls LLM (streaming), executes tools, and yields results."""
        logger.info(f"Processing input: '{user_input if not file_identifier else f'Prompt for file {file_identifier}'}'")
        yield {"type": "status", "data": "Processing input..."} # Status update

        user_message_content = []
        if file_identifier:
             file_part = process_file_input(file_identifier)
             if not file_part:
                  yield {"type": "error", "data": "[Agent Error: Could not process the provided file identifier.]"}
                  return
             user_message_content.extend([{"type": "text", "text": user_input}, file_part])
        else:
             user_message_content.append({"type": "text", "text": user_input})

        user_message = {"role": "user", "content": user_message_content}
        if not self.memory.add_message(user_message):
            yield {"type": "error", "data": "[Agent Error: Message too long for context.]"}
            return

        # Store user input text in VDB
        if self.vector_db.is_ready():
             self.vector_db.add(f"User said: {user_input}", {"type": "user_message", "time": datetime.now().isoformat()})

        yield {"type": "status", "data": "Thinking..."}
        logger.info("OmniBot: Thinking...")

        try:
            current_messages = self.memory.get_messages()
            response_stream = litellm.completion(
                model=self.model_name, messages=current_messages,
                tools=self.active_tool_schemas, tool_choice="auto", stream=True
            )

            # --- Process First Stream ---
            aggregated_response = defaultdict(lambda: "")
            aggregated_tool_calls = defaultdict(lambda: {"id": None, "name": None, "arguments": ""})
            completed_tool_indices = set()
            current_tool_idx = -1

            async for chunk in response_stream: # Use async for with stream
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"type": "content", "data": delta.content}
                    aggregated_response["content"] += delta.content
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx > current_tool_idx: current_tool_idx = idx
                        if tc_chunk.id: aggregated_tool_calls[idx]["id"] = tc_chunk.id
                        if tc_chunk.function and tc_chunk.function.name: aggregated_tool_calls[idx]["name"] = tc_chunk.function.name
                        if tc_chunk.function and tc_chunk.function.arguments: aggregated_tool_calls[idx]["arguments"] += tc_chunk.function.arguments
                        # Simple completion check (can be improved)
                        call_data = aggregated_tool_calls[idx]
                        if call_data["id"] and call_data["name"] and idx not in completed_tool_indices:
                            args_str = call_data["arguments"]; is_complete = False
                            try: json.loads(args_str); is_complete = True
                            except: pass
                            if is_complete: completed_tool_indices.add(idx)

            # Reconstruct first response message dict after stream ends
            final_tool_calls_list = []
            for idx in sorted(aggregated_tool_calls.keys()):
                 if idx in completed_tool_indices:
                     call = aggregated_tool_calls[idx]
                     final_tool_calls_list.append({"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}})
            response_message_dict = {"role": "assistant", "content": aggregated_response["content"] or None, "tool_calls": final_tool_calls_list or None}
            self.memory.add_message(response_message_dict) # Add aggregated first response

            # --- Handle Tool Calls ---
            if response_message_dict.get("tool_calls"):
                yield {"type": "status", "data": f"Using {len(response_message_dict['tool_calls'])} tool(s)..."}
                logger.info(f"LLM requested {len(response_message_dict['tool_calls'])} tool(s)...")

                tool_results_messages = []
                for tc_data in response_message_dict["tool_calls"]:
                    result_content = execute_tool_call(tc_data) # Execute using the dict
                    if isinstance(result_content, str) and result_content.lower().startswith("error"):
                         logger.warning(f"Tool '{tc_data.get('function', {}).get('name')}' failed. Error: {result_content}")
                         yield {"type": "tool_error", "data": {"tool_name": tc_data.get('function', {}).get('name'), "error": result_content}}
                    else:
                         yield {"type": "tool_result", "data": {"tool_name": tc_data.get('function', {}).get('name'), "result_preview": str(result_content)[:100] + "..."}}

                    result_msg = {"role": "tool", "tool_call_id": tc_data.get('id'), "content": str(result_content)}
                    tool_results_messages.append(result_msg)
                    self.memory.add_message(result_msg) # Add result to memory

                yield {"type": "status", "data": "Processing tool results..."}
                logger.info("OmniBot: Processing tool results...")
                messages_with_results = self.memory.get_messages()

                final_response_stream = litellm.completion(
                    model=self.model_name, messages=messages_with_results, stream=True
                )

                # --- Process Final Stream ---
                final_aggregated_content = ""
                async for chunk in final_response_stream:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        yield {"type": "content", "data": delta_content}
                        final_aggregated_content += delta_content

                final_response_dict = {"role": "assistant", "content": final_aggregated_content or None, "tool_calls": None} # No tool calls expected here
                self.memory.add_message(final_response_dict) # Add final assistant response

                if final_aggregated_content and self.vector_db.is_ready():
                    self.vector_db.add(f"OmniBot response: {final_aggregated_content}", {"type": "assistant_response", "after_tool_use": True, "time": datetime.now().isoformat()})
                logger.info(f"OmniBot final response (after tools): {final_aggregated_content or '[No Content]'}")

            # --- Handle Regular Text Response (no tool calls) ---
            elif response_message_dict.get("content"):
                assistant_content = response_message_dict["content"]
                # Content was already yielded by the first stream
                logger.info(f"OmniBot initial response (no tools): {assistant_content}")
                if self.vector_db.is_ready():
                    self.vector_db.add(f"OmniBot response: {assistant_content}", {"type": "assistant_response", "after_tool_use": False, "time": datetime.now().isoformat()})
            else:
                logger.warning("OmniBot received empty initial response (no content or tools).")
                yield {"type": "status", "data": "Done."} # Send a final status if nothing else was yielded

        except litellm.exceptions.APIError as e:
             logger.error(f"LiteLLM API Error: {e}", exc_info=True)
             yield {"type": "error", "data": f"[Agent Error: API communication failed. Status: {e.status_code if hasattr(e, 'status_code') else 'N/A'}]"}
        except AgentException as e:
             logger.error(f"Agent Error: {e}", exc_info=True)
             yield {"type": "error", "data": f"[Agent Error: {e}]"}
        except Exception as e:
             logger.critical(f"Unexpected critical error in agent processing!", exc_info=True)
             yield {"type": "error", "data": f"[Agent Critical Error: {e}]"}

        yield {"type": "status", "data": "Done"} # Signal end of processing for this turn

# Example of how to instantiate and use (this part won't run directly, used by app.py)
# if __name__ == "__main__":
#     agent = OmniBotAgent()
#     # Example usage (replace with actual API call in app.py)
#     async def run_test():
#         async for update in agent.process_message_stream("Hello there!"):
#             print(update)
#     import asyncio
#     asyncio.run(run_test())