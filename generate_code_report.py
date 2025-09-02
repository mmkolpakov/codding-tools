#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import concurrent.futures
import copy
import importlib.util
import io
import json
import logging
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import tkinter as tk
import tokenize
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import List, Optional, Tuple, Dict, Union, Type, Callable

# --- Dependency Check ---
def check_dependencies() -> List[str]:
    # (module_to_check, package_to_install)
    required = [
        ("tiktoken", "tiktoken"),
        ("charset_normalizer", "charset_normalizer"),
        ("pygments", "pygments"),
        ("TKinterModernThemes", "TKinterModernThemes"),
        ("git", "GitPython")
    ]
    missing = []
    for module, package in required:
        if not importlib.util.find_spec(module):
            missing.append(package)
    return missing


missing_deps = check_dependencies()
if missing_deps:
    msg = f"Missing required libraries: {', '.join(missing_deps)}.\nPlease run: pip install {' '.join(missing_deps)}"
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Dependency Error", msg)
    except tk.TclError:
        print(msg, file=sys.stderr)
    sys.exit(1)

# --- Third-Party Imports ---
import tiktoken
import git
from charset_normalizer import from_bytes
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound
import TKinterModernThemes as tkmt


# --- Constants & Enums ---
class SourceType(Enum):
    LOCAL = "Local"
    GIT = "Git"


class UIMode(Enum):
    DARK = "dark"
    LIGHT = "light"


APP_NAME = "Code Report Generator"
UI_THEME_NAME = "sun-valley"
LOG_FILE = Path("debug.log")
CONFIG_FILE = Path("config.json")
REPORT_HEADER_TEXT = "# Unified Code Report"
DEFAULT_MODEL = "Gemini 1.5 Pro"
HANDLER_NAME_FILE = "CRG_FileHandler"
HANDLER_NAME_UI = "CRG_UIHandler"
BIN_DETECT_CHUNK_SIZE = 8192
TOKEN_CACHE_SIZE = 8192

EXCLUDED_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', 'vendor', 'build', 'dist', '.venv'}
BUILTIN_MODEL_CONTEXT_SIZES = {
    'gpt-4o': 128000, 'gpt-4-turbo': 128000, 'gpt-4': 32768,
    'gpt-3.5-turbo-16k': 16385, DEFAULT_MODEL: 1000000,
    'Claude 3 Opus': 200000, 'Claude 3 Sonnet': 200000,
}


# --- Data Structures ---
@dataclass
class SourceConfig:
    path: str
    type: str
    branch: Optional[str] = None


@dataclass
class FileInfo:
    file_path: Path
    language: str
    code_content: str
    token_count: int


@dataclass
class ProjectInfo:
    project_name: str
    project_path: Path
    structure: str
    files: List[FileInfo]


@dataclass
class SourceData:
    project_name: str
    project_path: Path
    files_to_process: List[Path]
    structure_text: str


@dataclass
class AppConfig:
    sources: List[SourceConfig] = field(default_factory=list)
    unified_output_file: str = "unified_report.md"
    extensions: List[str] = field(default_factory=lambda: ['.py', '.js', '.java', '.go', '.rs', '.html', '.css'])
    include_libraries: bool = False
    remove_comments: bool = True
    preserve_docstrings: bool = True
    log_level: str = "INFO"
    selected_model: str = DEFAULT_MODEL
    custom_models: Dict[str, int] = field(default_factory=dict)
    max_threads: int = min(4, os.cpu_count() or 1)
    max_file_size_mb: int = 50
    ui_theme: str = UIMode.DARK.value


# --- UI Event Queue Payloads ---
@dataclass
class StatusUpdate:
    message: str


@dataclass
class ProgressUpdate:
    current: int
    total: int


@dataclass
class LogMessage:
    level: int
    message: str


@dataclass
class TaskFinished:
    success: bool


UIEvent = Union[StatusUpdate, ProgressUpdate, LogMessage, TaskFinished]

# --- Setup ---
logger = logging.getLogger(APP_NAME)
ui_event_queue: queue.Queue[UIEvent] = queue.Queue(maxsize=5000)


def setup_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if any(h.name == HANDLER_NAME_FILE for h in logger.handlers):
        for handler in logger.handlers:
            if handler.name in {HANDLER_NAME_FILE, HANDLER_NAME_UI}:
                handler.setLevel(log_level)
        return

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    try:
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        file_handler.name = HANDLER_NAME_FILE
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as exc:
        print(f"Could not create log file handler: {exc}", file=sys.stderr)

    class UIQueueLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if not ui_event_queue.full():
                ui_event_queue.put(LogMessage(level=record.levelno, message=self.format(record)))

    queue_handler = UIQueueLogHandler()
    queue_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))
    queue_handler.setLevel(log_level)
    queue_handler.name = HANDLER_NAME_UI
    logger.addHandler(queue_handler)


setup_logging("INFO")

try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as exc:
    logger.critical(f"Failed to initialize tiktoken encoder: {exc}", exc_info=True)
    sys.exit(1)


@lru_cache(maxsize=TOKEN_CACHE_SIZE)
def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text, disallowed_special=()))


# --- Core Logic ---
class CodeProcessor:
    @staticmethod
    def is_binary(file_path: Path) -> bool:
        try:
            with file_path.open('rb') as f:
                chunk = f.read(BIN_DETECT_CHUNK_SIZE)
            if not chunk: return False
            if b'\0' in chunk: return True
            if len(chunk) < 256: return False

            non_text_chars = sum(1 for byte in chunk if byte < 9 or (13 < byte < 32 and byte != 27) or byte > 127)
            return non_text_chars / len(chunk) > 0.3
        except (IOError, OSError) as exc:
            logger.warning(f"Could not check file type for {file_path}: {exc}", exc_info=True)
            return True

    @staticmethod
    def detect_encoding(file_path: Path, max_size_mb: int) -> Optional[str]:
        try:
            file_size = file_path.stat().st_size
            if file_size > max_size_mb * 1024 * 1024:
                logger.warning(f"File {file_path} is too large (> {max_size_mb}MB), skipping.")
                return None

            read_size = min(file_size, 256 * 1024)
            with file_path.open('rb') as f:
                raw_data = f.read(read_size)
            result = from_bytes(raw_data).best()
            return result.encoding if result else 'utf-8'
        except (IOError, OSError) as exc:
            logger.error(f"Error detecting encoding for {file_path}: {exc}", exc_info=True)
            return None

    @staticmethod
    def read_file_safely(file_path: Path, encoding: str) -> Optional[str]:
        try:
            return file_path.read_text(encoding=encoding, errors='replace')
        except (IOError, OSError) as exc:
            logger.error(f"Error reading file {file_path}: {exc}", exc_info=True)
            return None

    @staticmethod
    def remove_python_comments(source: str, preserve_docstrings: bool) -> str:
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
            if preserve_docstrings:
                return tokenize.untokenize(t for t in tokens if t.type != tokenize.COMMENT)

            result_tokens = []
            for i, token in enumerate(tokens):
                if token.type == tokenize.COMMENT: continue
                if token.type == tokenize.STRING and i > 0 and tokens[i - 1].type in {tokenize.NEWLINE, tokenize.NL,
                                                                                      tokenize.INDENT}:
                    continue
                result_tokens.append(token)
            return tokenize.untokenize(result_tokens)
        except (tokenize.TokenError, IndentationError) as exc:
            logger.warning(f"Python tokenization failed: {exc}. Falling back to basic regex.")
            return re.sub(r'#.*', '', source)

    @staticmethod
    def remove_comments(file_path: Path, source: str, preserve_docstrings: bool) -> str:
        if file_path.suffix.lower() == '.py':
            return CodeProcessor.remove_python_comments(source, preserve_docstrings)
        pattern = r"//.*?$|/\*.*?\*/|--.*?$"
        return re.sub(pattern, "", source, flags=re.MULTILINE | re.DOTALL)

    @staticmethod
    def detect_language(file_path: Path, content: str) -> str:
        try:
            lexer = guess_lexer_for_filename(file_path.name, content)
            return lexer.aliases[0] if lexer.aliases else lexer.name.lower()
        except ClassNotFound:
            return file_path.suffix.lstrip('.').lower() or 'text'

    def process_file(self, file_path: Path, config: AppConfig) -> Optional[FileInfo]:
        encoding = self.detect_encoding(file_path, config.max_file_size_mb)
        if not encoding: return None

        content = self.read_file_safely(file_path, encoding)
        if content is None: return None

        if config.remove_comments:
            content = self.remove_comments(file_path, content, config.preserve_docstrings)

        return FileInfo(
            file_path=file_path,
            language=self.detect_language(file_path, content),
            code_content=content.strip(),
            token_count=count_tokens(content)
        )


class SourceProvider(ABC):
    def __init__(self, source_config: SourceConfig, ui_queue: queue.Queue):
        self.config = source_config
        self.ui_queue = ui_queue
        self.temp_dir: Optional[Path] = None

    @abstractmethod
    def prepare(self) -> Optional[Path]:
        """Prepares the source (e.g., clones repo, checks path) and returns the path to it."""
        pass

    @staticmethod
    def _build_structure_text(root: Path, dirs: set) -> str:
        lines = ["### Project Structure:\n"]
        sorted_paths = sorted(list(p for p in dirs if p != root), key=lambda p: p.as_posix())

        for path in sorted_paths:
            try:
                relative_parts = path.relative_to(root).parts
                level = len(relative_parts)
                lines.append(f"{'  ' * (level - 1)}- {path.name}/\n")
            except ValueError:
                continue
        return "".join(lines) + "\n"

    def gather_data(self, project_path: Path, extensions: set, include_libs: bool,
                      is_binary_checker: Callable[[Path], bool]) -> SourceData:
        files_to_scan = []
        structure_dirs = set()
        use_wildcard = "*" in extensions

        for dir_path, dirs, files in os.walk(project_path, topdown=True):
            if not include_libs:
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('.')]

            for file_name in files:
                file_path = Path(dir_path) / file_name
                if use_wildcard or file_path.suffix.lower() in extensions:
                    if not is_binary_checker(file_path):
                        files_to_scan.append(file_path)
                        structure_dirs.add(Path(dir_path))

        structure_text = self._build_structure_text(project_path, structure_dirs)
        return SourceData(
            project_name=project_path.name,
            project_path=project_path,
            files_to_process=sorted(files_to_scan),
            structure_text=structure_text
        )

    def cleanup(self):
        """Cleans up temporary resources."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")


class LocalSourceProvider(SourceProvider):
    def prepare(self) -> Optional[Path]:
        path = Path(self.config.path)
        if not path.is_dir():
            logger.error(f"Local path is not a valid directory: {path}")
            self.ui_queue.put(StatusUpdate(message=f"Error: Path not found {path}"))
            return None
        return path


class GitSourceProvider(SourceProvider):
    def prepare(self) -> Optional[Path]:
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="crg_git_"))
            branch_info = f" (branch: {self.config.branch})" if self.config.branch else ""
            self.ui_queue.put(StatusUpdate(message=f"Cloning {self.config.path}{branch_info}..."))

            clone_kwargs = {'depth': 1}
            if self.config.branch:
                clone_kwargs['branch'] = self.config.branch

            git.Repo.clone_from(self.config.path, self.temp_dir, **clone_kwargs)
            self.ui_queue.put(StatusUpdate(message="Clone successful."))
            return self.temp_dir
        except git.GitCommandError as exc:
            logger.error(f"Failed to clone {self.config.path}: {exc}", exc_info=True)
            self.ui_queue.put(StatusUpdate(message=f"Error cloning {self.config.path}."))
            self.cleanup()
            return None


class ReportGenerator:
    def __init__(self, max_tokens: int, cancel_flag: threading.Event):
        self.max_tokens = max_tokens
        self.cancel_flag = cancel_flag
        self.report_header_full = f"{REPORT_HEADER_TEXT}\n\n"
        self.report_header_tokens = count_tokens(self.report_header_full)

    def generate(self, all_projects: List[ProjectInfo]) -> List[str]:
        report_parts = []
        current_part = [self.report_header_full]
        current_tokens = self.report_header_tokens

        for project in all_projects:
            if self.cancel_flag.is_set(): break

            project_header = f"## Project: {project.project_name}\n\n{project.structure}"
            project_header_tokens = count_tokens(project_header)

            if current_tokens > self.report_header_tokens and current_tokens + project_header_tokens > self.max_tokens:
                report_parts.append("".join(current_part))
                current_part = [self.report_header_full, project_header]
                current_tokens = self.report_header_tokens + project_header_tokens
            else:
                current_part.append(project_header)
                current_tokens += project_header_tokens

            for file_info in project.files:
                if self.cancel_flag.is_set(): break
                current_tokens = self._add_file_to_report(file_info, project, report_parts, current_part,
                                                          current_tokens)

        if current_part and "".join(current_part).strip() != self.report_header_full.strip():
            report_parts.append("".join(current_part))

        return report_parts

    def _add_file_to_report(self, file_info: FileInfo, project: ProjectInfo, report_parts: List, current_part: List,
                            current_tokens: int) -> int:
        relative_path = file_info.file_path.relative_to(project.project_path)
        file_header = f"#### {relative_path}\n\n```{file_info.language}\n"
        file_footer = "\n```\n\n"
        wrapper_tokens = count_tokens(file_header + file_footer)

        project_header = f"## Project: {project.project_name}\n\n{project.structure}"
        project_header_tokens = count_tokens(project_header)

        # Check if the file itself is smaller than the total context window
        if file_info.token_count + wrapper_tokens + self.report_header_tokens + project_header_tokens < self.max_tokens:
            file_section = f"{file_header}{file_info.code_content}{file_footer}"

            # If adding this file exceeds the current part's limit, start a new part
            if current_tokens > self.report_header_tokens and current_tokens + file_info.token_count + wrapper_tokens > self.max_tokens:
                report_parts.append("".join(current_part))
                current_part.clear()
                current_part.extend([self.report_header_full, project_header])
                current_tokens = self.report_header_tokens + project_header_tokens

            current_part.append(file_section)
            return current_tokens + file_info.token_count + wrapper_tokens
        else:
            # File is too large, needs chunking. Finish the current part first.
            if len(current_part) > 1:
                report_parts.append("".join(current_part))
            current_part.clear() # Prepare for the next part after chunking

            logger.warning(f"File {relative_path} is too large and will be chunked across multiple report parts.")
            lines = file_info.code_content.splitlines(keepends=True)
            chunk_lines = []
            chunk_tokens = 0

            part_header = f"{self.report_header_full}{project_header}"
            part_header_tokens = self.report_header_tokens + project_header_tokens

            for line in lines:
                if self.cancel_flag.is_set(): break
                line_tokens = count_tokens(line)
                if chunk_tokens + line_tokens + wrapper_tokens + part_header_tokens > self.max_tokens:
                    report_parts.append(f"{part_header}{file_header}{''.join(chunk_lines)}{file_footer}")
                    chunk_lines, chunk_tokens = [], 0
                chunk_lines.append(line)
                chunk_tokens += line_tokens
            if chunk_lines:
                report_parts.append(f"{part_header}{file_header}{''.join(chunk_lines)}{file_footer}")

            current_part.extend([self.report_header_full])
            return self.report_header_tokens


class ProjectProcessor:
    def __init__(self, config: AppConfig, code_processor: CodeProcessor,
                 executor: concurrent.futures.Executor, ui_queue: queue.Queue[UIEvent]):
        self.config = config
        self.code_processor = code_processor
        self.executor = executor
        self.ui_queue = ui_queue
        self.cancel_flag = threading.Event()
        self.source_providers_map: Dict[str, Type[SourceProvider]] = {
            SourceType.LOCAL.name: LocalSourceProvider,
            SourceType.GIT.name: GitSourceProvider,
        }

    def cancel(self):
        self.cancel_flag.set()
        self.ui_queue.put(StatusUpdate(message="Cancellation requested by user."))

    def run(self) -> None:
        all_projects_info: List[ProjectInfo] = []
        active_providers: List[SourceProvider] = []

        try:
            total_projects = len(self.config.sources)
            for i, source_config in enumerate(self.config.sources, 1):
                if self.cancel_flag.is_set(): break
                self.ui_queue.put(StatusUpdate(message=f"Processing project {i}/{total_projects}: {source_config.path}"))

                provider_class = self.source_providers_map.get(source_config.type)
                if not provider_class:
                    logger.error(f"Invalid source type in config: {source_config.type}")
                    continue

                provider = provider_class(source_config, self.ui_queue)
                active_providers.append(provider)

                project_path = provider.prepare()
                if not project_path: continue

                source_data = provider.gather_data(project_path, set(self.config.extensions),
                                                   self.config.include_libraries, self.code_processor.is_binary)

                if not source_data.files_to_process: continue

                processed_files: List[FileInfo] = []
                futures = {self.executor.submit(self.code_processor.process_file, fp, self.config): fp for fp in
                           source_data.files_to_process}

                total_files = len(source_data.files_to_process)
                for processed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    if self.cancel_flag.is_set(): break
                    try:
                        result = future.result()
                        if result: processed_files.append(result)
                    except Exception as exc:
                        logger.error(f"Error processing file {futures[future]}: {exc}", exc_info=True)
                    self.ui_queue.put(ProgressUpdate(current=processed_count, total=total_files))

                if self.cancel_flag.is_set(): break

                all_projects_info.append(ProjectInfo(
                    project_name=source_data.project_name, project_path=project_path,
                    structure=source_data.structure_text, files=sorted(processed_files, key=lambda f: f.file_path)
                ))

            if not self.cancel_flag.is_set():
                models = {**BUILTIN_MODEL_CONTEXT_SIZES, **self.config.custom_models}
                max_tokens = models.get(self.config.selected_model, 128000)
                report_generator = ReportGenerator(max_tokens, self.cancel_flag)

                self.ui_queue.put(StatusUpdate(message="Generating report parts..."))
                report_parts = report_generator.generate(all_projects_info)
                self._save_reports(report_parts)

            self.ui_queue.put(TaskFinished(success=not self.cancel_flag.is_set()))

        except Exception as exc:
            logger.critical(f"Unhandled exception in processor thread: {exc}", exc_info=True)
            self.ui_queue.put(TaskFinished(success=False))
        finally:
            for provider in active_providers:
                provider.cleanup()

    def _save_reports(self, parts: List[str]):
        if not parts:
            logger.warning("No content generated for reports.")
            return

        out_file = Path(self.config.unified_output_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        for i, content in enumerate(parts):
            if self.cancel_flag.is_set(): break
            path = out_file if len(parts) == 1 else out_file.with_name(f"{out_file.stem}_part{i + 1}{out_file.suffix}")
            try:
                path.write_text(content, encoding='utf-8', newline='\n')
                logger.info(f"Report part saved to {path}")
            except IOError as exc:
                logger.error(f"Failed to save report part to {path}: {exc}", exc_info=True)


# --- GUI ---
class ConfigManager:
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.config = AppConfig()
        self.load_config()

    def load_config(self) -> None:
        if not self.config_file.exists(): return
        try:
            with self.config_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            default_dict = asdict(AppConfig())
            merged_data = {**default_dict, **data}

            sources_data = merged_data.get("sources", [])
            merged_data["sources"] = []
            for s_item in sources_data:
                # Handle old configs without 'branch'
                if 'branch' not in s_item:
                    s_item['branch'] = None
                merged_data["sources"].append(SourceConfig(**s_item))

            valid_keys = {f.name for f in fields(AppConfig)}
            filtered_data = {k: v for k, v in merged_data.items() if k in valid_keys}

            self.config = AppConfig(**filtered_data)
        except (json.JSONDecodeError, IOError, TypeError) as exc:
            logger.error(f"Error loading or parsing config file, using defaults: {exc}", exc_info=True)
            self.config = AppConfig()

    def save_config(self, config: AppConfig) -> None:
        self.config = config
        try:
            with self.config_file.open("w", encoding="utf-8") as f:
                json.dump(asdict(config), f, indent=4)
        except (IOError, TypeError) as exc:
            logger.error(f"Error saving config: {exc}", exc_info=True)


class App(tkmt.ThemedTKinterFrame):
    def __init__(self, config_manager: ConfigManager, theme: str, mode: str):
        super().__init__(APP_NAME, theme, mode)
        self.config_manager = config_manager
        self.config = copy.deepcopy(config_manager.config)
        self.processor: Optional[ProjectProcessor] = None
        self.executor: Optional[concurrent.futures.Executor] = None
        self.is_processing = False
        self.log_buffer: List[LogMessage] = []
        self.log_buffer_timer: Optional[str] = None
        self._log_tags_configured = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.minsize(900, 750)
        self.center_window()

        self.create_widgets()
        self.load_config_into_ui()
        self.process_ui_queue()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)

        settings_pane = ttk.Frame(main_pane)
        log_pane = ttk.Frame(main_pane)
        main_pane.add(settings_pane, weight=1)
        main_pane.add(log_pane, weight=2)

        self._create_settings_ui(settings_pane)
        self._create_log_ui(log_pane)

    def _create_settings_ui(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, pady=5)

        tabs = {"Sources": ttk.Frame(notebook), "File Options": ttk.Frame(notebook), "Advanced": ttk.Frame(notebook)}
        for name, frame in tabs.items(): notebook.add(frame, text=name)

        self._create_sources_tab(tabs["Sources"])
        self._create_file_options_tab(tabs["File Options"])
        self._create_advanced_tab(tabs["Advanced"])

        action_frame = ttk.Frame(parent)
        action_frame.pack(fill="x", pady=10)
        self.btn_start = ttk.Button(action_frame, text="Start Processing", command=self.start_processing)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=5)
        self.btn_cancel = ttk.Button(action_frame, text="Cancel", command=self.cancel_processing, state="disabled")
        self.btn_cancel.pack(side="left", expand=True, fill="x", padx=5)

    def _create_sources_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.source_tree = ttk.Treeview(frame, columns=("type", "path", "branch"), show="headings", selectmode="browse")
        self.source_tree.heading("type", text="Type", anchor="w")
        self.source_tree.heading("path", text="Path / URL", anchor="w")
        self.source_tree.heading("branch", text="Branch", anchor="w")
        self.source_tree.column("type", width=80, stretch=tk.NO)
        self.source_tree.column("branch", width=100, stretch=tk.NO)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.source_tree.yview)
        self.source_tree.configure(yscrollcommand=scrollbar.set)

        self.source_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="Add Directory...", command=self.add_local_directory).pack(side="left", expand=True,
                                                                                              fill="x", padx=2)
        ttk.Button(btn_frame, text="Add Git URL...", command=self.add_git_url).pack(side="left", expand=True, fill="x",
                                                                                    padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected_source).pack(side="left",
                                                                                                expand=True, fill="x",
                                                                                                padx=2)

    def _create_file_options_tab(self, parent):
        self.extensions_var = tk.StringVar()
        self.include_libs_var = tk.BooleanVar()
        self.remove_comments_var = tk.BooleanVar()
        self.preserve_docstrings_var = tk.BooleanVar()
        self.unified_output_var = tk.StringVar()

        ttk.Label(parent, text="File Extensions (comma-separated, * for all):").pack(anchor="w", padx=5, pady=(5, 0))
        ttk.Entry(parent, textvariable=self.extensions_var).pack(fill="x", padx=5, pady=2)

        ttk.Label(parent, text="Unified Report Filename:").pack(anchor="w", padx=5, pady=(10, 0))
        ttk.Entry(parent, textvariable=self.unified_output_var).pack(fill="x", padx=5, pady=2)

        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding=5)
        options_frame.pack(fill="x", padx=5, pady=10)

        ttk.Checkbutton(options_frame, text="Include hidden/library directories", variable=self.include_libs_var).pack(
            anchor="w")
        ttk.Checkbutton(options_frame, text="Remove comments from code", variable=self.remove_comments_var,
                        command=self.sync_docstring_option_state).pack(anchor="w")
        self.cb_preserve_docstrings = ttk.Checkbutton(options_frame, text="Preserve docstrings (if removing comments)",
                                                      variable=self.preserve_docstrings_var)
        self.cb_preserve_docstrings.pack(anchor="w", padx=20)

    def _create_advanced_tab(self, parent):
        self.selected_model_var = tk.StringVar()
        self.log_level_var = tk.StringVar()
        self.max_threads_var = tk.IntVar()
        self.max_file_size_var = tk.IntVar()

        model_frame = ttk.LabelFrame(parent, text="Model and Performance", padding=5)
        model_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(model_frame, text="Model for Token Limit:").pack(anchor="w")
        self.combo_model = ttk.Combobox(model_frame, textvariable=self.selected_model_var, state="readonly")
        self.combo_model.pack(fill="x", pady=2)

        manage_models_btn = ttk.Button(model_frame, text="Manage Custom Models...", state="disabled")
        manage_models_btn.pack(fill="x", pady=(5, 0))
        if hasattr(tkmt, "CreateToolTip"):
            tkmt.CreateToolTip(manage_models_btn, "This feature is not yet implemented.")

        cpu_count = os.cpu_count() or 1
        ttk.Label(model_frame, text=f"Max Processing Threads (1-{cpu_count}):").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(model_frame, from_=1, to=cpu_count, textvariable=self.max_threads_var).pack(fill="x", pady=2)

        ttk.Label(model_frame, text="Max File Size to Process (MB):").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(model_frame, from_=1, to=200, textvariable=self.max_file_size_var).pack(fill="x", pady=2)

        logging_frame = ttk.LabelFrame(parent, text="Logging", padding=5)
        logging_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(logging_frame, text="Log Level:").pack(anchor="w")
        ttk.Combobox(logging_frame, textvariable=self.log_level_var,
                     values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], state="readonly").pack(fill="x", pady=2)

    def _create_log_ui(self, parent):
        ttk.Label(parent, text="Logs & Progress", font=("", 12, "bold")).pack(fill="x", pady=(0, 10))
        self.log_text = scrolledtext.ScrolledText(parent, wrap='none', state='disabled', height=10)
        self.log_text.pack(fill='both', expand=True, pady=5)

        is_dark_theme = self.config.ui_theme == UIMode.DARK.value
        self.log_text.tag_config("ERROR", foreground="#FF6B6B" if is_dark_theme else "#D8000C")
        self.log_text.tag_config("CRITICAL", foreground="#FF6B6B" if is_dark_theme else "#D8000C", font=("", 9, "bold"))
        self.log_text.tag_config("WARNING", foreground="#FFD166" if is_dark_theme else "#9F6000")
        self.log_text.tag_config("INFO", foreground="#A9A9A9" if is_dark_theme else "#555555")
        self.log_text.tag_config("DEBUG", foreground="#6A7A8A" if is_dark_theme else "#888888")
        self._log_tags_configured = True

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(parent, textvariable=self.status_var, wraplength=400).pack(fill="x", padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var)
        self.progress_bar.pack(fill="x", padx=5, pady=5)

    def load_config_into_ui(self):
        self.extensions_var.set(", ".join(self.config.extensions))
        self.include_libs_var.set(self.config.include_libraries)
        self.remove_comments_var.set(self.config.remove_comments)
        self.preserve_docstrings_var.set(self.config.preserve_docstrings)
        self.unified_output_var.set(self.config.unified_output_file)
        self.selected_model_var.set(self.config.selected_model)
        self.log_level_var.set(self.config.log_level)
        self.max_threads_var.set(self.config.max_threads)
        self.max_file_size_var.set(self.config.max_file_size_mb)

        self.update_model_combobox()
        self.populate_source_tree()
        self.sync_docstring_option_state()

    def gather_ui_into_config(self):
        self.config.extensions = [e.strip() for e in self.extensions_var.get().split(',') if e.strip()]
        self.config.include_libraries = self.include_libs_var.get()
        self.config.remove_comments = self.remove_comments_var.get()
        self.config.preserve_docstrings = self.preserve_docstrings_var.get()
        self.config.unified_output_file = self.unified_output_var.get()
        self.config.selected_model = self.selected_model_var.get()
        self.config.log_level = self.log_level_var.get()
        self.config.max_threads = self.max_threads_var.get()
        self.config.max_file_size_mb = self.max_file_size_var.get()

    def start_processing(self):
        if self.is_processing: return
        self.gather_ui_into_config()
        self.config_manager.save_config(self.config)

        if not self.config.sources:
            messagebox.showwarning("Input Required", "Please add at least one source directory or Git URL.")
            return

        self.set_processing_state(True)
        setup_logging(self.config.log_level)

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_threads,
                                                              thread_name_prefix="Worker")
        self.processor = ProjectProcessor(
            config=self.config, code_processor=CodeProcessor(),
            executor=self.executor, ui_queue=ui_event_queue
        )
        threading.Thread(target=self.processor.run, daemon=True).start()

    def cancel_processing(self):
        self.status_var.set("Cancelling...")
        if self.processor: self.processor.cancel()
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="disabled")

    def set_processing_state(self, is_processing: bool):
        self.is_processing = is_processing
        self.btn_start.config(state="disabled" if is_processing else "normal")
        self.btn_cancel.config(state="normal" if is_processing else "disabled")
        if is_processing:
            self.progress_bar.config(mode="determinate")
            self.progress_var.set(0)
            self.status_var.set("Starting...")
        else:
            self.progress_var.set(0)

    def process_ui_queue(self):
        try:
            while not ui_event_queue.empty():
                event = ui_event_queue.get_nowait()
                if isinstance(event, StatusUpdate):
                    self.status_var.set(event.message)
                elif isinstance(event, ProgressUpdate):
                    self.progress_var.set((event.current / event.total) * 100 if event.total > 0 else 0)
                elif isinstance(event, LogMessage):
                    self.log_buffer.append(event)
                elif isinstance(event, TaskFinished):
                    self.set_processing_state(False)
                    if self.executor:
                        self.executor.shutdown(wait=False)
                        self.executor = None
                    self.processor = None
                    self.status_var.set(
                        "Processing finished." if event.success else "Processing failed or was cancelled.")
        except queue.Empty:
            pass
        finally:
            if not self.log_buffer_timer and self.log_buffer:
                self.log_buffer_timer = self.root.after(100, self.flush_log_buffer)
            self.root.after(100, self.process_ui_queue)

    def flush_log_buffer(self):
        self.log_buffer_timer = None
        if not self.log_buffer: return

        self.log_text.configure(state='normal')
        if not self._log_tags_configured:
            self._create_log_ui(self.log_text.master)

        for log_event in self.log_buffer:
            level_name = logging.getLevelName(log_event.level)
            tags = (level_name,) if level_name in self.log_text.tag_names() else ()
            self.log_text.insert(tk.END, log_event.message + '\n', tags)
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)
        self.log_buffer.clear()

    def populate_source_tree(self):
        self.source_tree.delete(*self.source_tree.get_children())
        for source in self.config.sources:
            self.source_tree.insert("", "end", values=(source.type, source.path, source.branch or ''))

    def add_local_directory(self):
        path = filedialog.askdirectory(title="Select Project Directory")
        if path and not any(s.path == path for s in self.config.sources):
            self.config.sources.append(SourceConfig(path=path, type=SourceType.LOCAL.name))
            self.populate_source_tree()

    def add_git_url(self):
        dialog = GitUrlDialog(self.root, "Add Git Repository")
        self.root.wait_window(dialog.top) # Wait for dialog to close

        if dialog.result:
            url, branch = dialog.result
            if url and (url.startswith("http://") or url.startswith("https://")):
                if not any(s.path == url and s.branch == branch for s in self.config.sources):
                    self.config.sources.append(SourceConfig(path=url, type=SourceType.GIT.name, branch=branch))
                    self.populate_source_tree()
            elif url:
                messagebox.showwarning("Invalid URL", "Please enter a valid HTTP/HTTPS URL.", parent=self.root)

    def remove_selected_source(self):
        selected_item = self.source_tree.selection()
        if not selected_item: return

        item_values = self.source_tree.item(selected_item[0])["values"]
        path_to_remove = item_values[1]
        branch_to_remove_str = item_values[2]
        branch_to_remove = branch_to_remove_str if branch_to_remove_str else None

        self.config.sources = [
            s for s in self.config.sources
            if not (s.path == path_to_remove and s.branch == branch_to_remove)
        ]
        self.populate_source_tree()

    def update_model_combobox(self):
        models = sorted(list(BUILTIN_MODEL_CONTEXT_SIZES.keys()) + list(self.config.custom_models.keys()))
        self.combo_model['values'] = models
        if self.selected_model_var.get() not in models:
            self.selected_model_var.set(DEFAULT_MODEL)

    def sync_docstring_option_state(self):
        state = "normal" if self.remove_comments_var.get() else "disabled"
        self.cb_preserve_docstrings.config(state=state)

    def on_close(self):
        if self.is_processing:
            if not messagebox.askyesno("Confirm Exit", "Processing is active. Are you sure you want to exit?"):
                return
            if self.processor: self.processor.cancel()
            if self.executor: self.executor.shutdown(wait=True)

        self.gather_ui_into_config()
        self.config_manager.save_config(self.config)

        if self.log_buffer_timer: self.root.after_cancel(self.log_buffer_timer)
        self.root.destroy()


class GitUrlDialog:
    def __init__(self, parent, title):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()

        self.url_var = tk.StringVar()
        self.branch_var = tk.StringVar()
        self.result: Optional[Tuple[str, Optional[str]]] = None

        ttk.Label(self.top, text="Repository URL (HTTPS):").pack(padx=10, pady=(10, 2))
        url_entry = ttk.Entry(self.top, textvariable=self.url_var, width=50)
        url_entry.pack(padx=10, pady=2, fill="x", expand=True)

        ttk.Label(self.top, text="Branch (optional, leave empty for default):").pack(padx=10, pady=(10, 2))
        ttk.Entry(self.top, textvariable=self.branch_var, width=50).pack(padx=10, pady=2, fill="x", expand=True)

        button_frame = ttk.Frame(self.top)
        button_frame.pack(padx=10, pady=10, fill="x")
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.top.destroy).pack(side="right")

        url_entry.focus_set()
        self.top.wait_visibility()
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.top.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.top.winfo_height() // 2)
        self.top.geometry(f"+{x}+{y}")


    def on_ok(self):
        url = self.url_var.get().strip()
        branch = self.branch_var.get().strip() or None
        self.result = (url, branch)
        self.top.destroy()


if __name__ == "__main__":
    config_manager = ConfigManager(CONFIG_FILE)
    setup_logging(config_manager.config.log_level)

    app = App(config_manager, theme=UI_THEME_NAME, mode=config_manager.config.ui_theme)
    app.run()