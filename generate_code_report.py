#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import json
import threading
import queue
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
import importlib.util
import time
import concurrent.futures
import io
import tokenize

REQUIRED_LIBRARIES = ["tiktoken", "charset_normalizer", "pygments"]
missing_libraries = []
for library in REQUIRED_LIBRARIES:
    if importlib.util.find_spec(library) is None:
        missing_libraries.append(library)
if missing_libraries:
    print(f"Required libraries not installed: {', '.join(missing_libraries)}")
    print(f"Install them with: pip install {' '.join(missing_libraries)}")
    sys.exit(1)

import tiktoken
from charset_normalizer import from_bytes
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound

try:
    import TKinterModernThemes as TKMT
except ImportError:
    print("TKinterModernThemes library is not installed")
    print("Install it with: pip install TKinterModernThemes")
    sys.exit(1)

DEFAULT_CONFIG = {
    "src_directories": [{"path": str(Path.home()), "selected": True}],
    "output_file_template": "{project_name}_report.md",
    "unified_output_file": "unified_report.md",
    "extensions": ['.py', '.kt', '.kts', '.cpp', '.hpp', '.h', '.cs', '.sv'],
    "include_libraries": False,
    "remove_comments": True,
    "preserve_docstrings": True,
    "log_level": "INFO",
    "selected_model": "Gemini",
    "unified_report": True,
    "custom_models": {},
    "max_threads": 4,
    "token_cache_size": 1000,
    "chunk_size": 1024,
    "max_file_size_mb": 50,
    "ui_theme": "dark"
}

BUILTIN_MODEL_CONTEXT_SIZES = {
    'gpt-3.5-turbo': 4096,
    'gpt-4': 32768,
    'gpt-4o': 128000,
    'o1-mini': 65536,
    'o1-preview': 25000,
    'Gemini': 2000000,
    'Claude': 200000,
    'LearnLm': 32767,
}

CONFIG_FILE = Path("config.json")


class ConfigManager:
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self) -> None:
        if self.config_file.exists():
            try:
                with self.config_file.open("r", encoding="utf-8") as file:
                    loaded_config = json.load(file)
                self._update_config_recursively(self.config, loaded_config)
                if self.config.get("src_directories") and isinstance(self.config["src_directories"][0], str):
                    self.config["src_directories"] = [
                        {"path": directory, "selected": True} for directory in self.config["src_directories"]
                    ]
            except json.JSONDecodeError as error:
                logging.error(f"Error reading JSON: {error}")
            except Exception as error:
                logging.error(f"Error loading configuration: {error}")

    def _update_config_recursively(self, target: Dict, source: Dict) -> None:
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._update_config_recursively(target[key], value)
            else:
                target[key] = value

    def save_config(self) -> None:
        try:
            with self.config_file.open("w", encoding="utf-8") as file:
                json.dump(self.config, file, indent=4, ensure_ascii=False)
        except Exception as error:
            logging.error(f"Error saving configuration: {error}")


config_manager = ConfigManager(CONFIG_FILE)

log_queue = queue.Queue()


class QueueHandler(logging.Handler):
    """Класс-обёртка для пересылки логов в очередь"""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            message = self.format(record)
            self.log_queue.put(message)
        except Exception:
            self.handleError(record)


logger = logging.getLogger("CodeReportGenerator")
logger.setLevel(
    getattr(logging, config_manager.config.get("log_level", "INFO").upper(), logging.INFO)
)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler("debug.log", encoding="utf-8", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as error:
    logger.critical(f"Failed to initialize tiktoken encoder: {error}")
    sys.exit(1)

def tokenize_text(text: str):
    return ENCODER.encode(text)

def count_tokens(text: str) -> int:
    return len(tokenize_text(text))

def count_tokens_in_chunks(text: str, chunk_size: int) -> int:
    total_tokens = 0
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        total_tokens += len(tokenize_text(chunk))
    return total_tokens


class CodeProcessor:
    def __init__(self):
        self.extension_to_language = {
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.sh': 'bash',
            '.bat': 'batch',
            '.ps1': 'powershell',
            '.sql': 'sql',
            '.md': 'markdown',
            '.json': 'json',
            '.xml': 'xml',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.sv': 'systemverilog'
        }

    def is_binary(self, file_path: Path) -> bool:
        """Простая эвристика для проверки, не двоичный ли файл"""
        try:
            with file_path.open('rb') as file:
                chunk = file.read(512)
                return (
                    b'\0' in chunk or
                    sum(1 for b in chunk if b < 8 or 127 < b < 192) > len(chunk) * 0.3
                )
        except (IOError, OSError) as error:
            logger.error(f"Error checking file {file_path}: {error}", exc_info=True)
            return True
        except Exception as error:
            logger.error(f"Unexpected error checking file {file_path}: {error}", exc_info=True)
            return True

    def detect_encoding(self, file_path: Path, max_file_size_mb: int) -> Optional[str]:
        """Определяем кодировку файла с помощью charset_normalizer"""
        try:
            file_size = file_path.stat().st_size
            if file_size > max_file_size_mb * 1024 * 1024:
                logger.warning(
                    f"File {file_path} is too large (> {max_file_size_mb} MB), will be skipped."
                )
                raise ValueError(f"File is too large (> {max_file_size_mb} MB)")

            with file_path.open('rb') as file:
                raw_data = file.read(min(file_size, 100000))

            result = from_bytes(raw_data).best()
            return result.encoding if result else 'utf-8'
        except (IOError, OSError) as error:
            logger.error(f"Error detecting encoding for file {file_path}: {error}", exc_info=True)
            return None
        except Exception as error:
            logger.error(f"Unexpected error detecting encoding for file {file_path}: {error}", exc_info=True)
            return None

    def read_file_safely(self, file_path: Path, encoding: str) -> Optional[str]:
        """Безопасно читаем файл, заменяя битые символы"""
        try:
            with file_path.open('r', encoding=encoding, errors='replace') as file:
                return file.read()
        except (IOError, OSError) as error:
            logger.error(f"Error reading file {file_path}: {error}", exc_info=True)
            return None
        except Exception as error:
            logger.error(f"Unexpected error reading file {file_path}: {error}", exc_info=True)
            return None

    def remove_python_comments(self, source_code: str, preserve_docstrings: bool = True) -> str:
        """
        Удаляем комментарии из Python-кода, при желании — docstrings.
        Логика покрывает большинство базовых случаев.
        """
        result = []
        try:
            tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)

            prev_token_type = None
            prev_token_string = None
            last_line = 1
            last_col = 0

            for token in tokens:
                token_type = token.type
                token_string = token.string
                start_line, start_col = token.start
                end_line, end_col = token.end

                if start_line > last_line:
                    result.append('\n' * (start_line - last_line))
                    last_col = 0

                if start_col > last_col:
                    result.append(' ' * (start_col - last_col))

                if token_type == tokenize.COMMENT:

                    pass
                elif token_type == tokenize.STRING:
                    is_docstring = (
                        prev_token_type == tokenize.INDENT
                        or prev_token_string in ('class', 'def')
                        or (prev_token_type == tokenize.NEWLINE and start_col == 0)
                    )
                    if is_docstring and not preserve_docstrings:
                        pass
                    else:
                        result.append(token_string)
                else:
                    result.append(token_string)

                prev_token_type = token_type
                prev_token_string = token_string
                last_line, last_col = end_line, end_col

            return ''.join(result)
        except tokenize.TokenError:
            return self._remove_comments_regex(source_code, preserve_docstrings)

    def _remove_comments_regex(self, source_code: str, preserve_docstrings: bool = True) -> str:
        """
        Простейший regex-based метод на случай проблем с tokenize
        """
        lines = source_code.splitlines()
        result = []
        in_multiline = False

        for line in lines:
            stripped = line.strip()

            if in_multiline:
                if '"""' in line or "'''" in line:
                    in_multiline = False
                    if preserve_docstrings:
                        result.append(line)
                elif preserve_docstrings:
                    result.append(line)
            else:
                if stripped.startswith('#'):
                    result.append('')
                elif '#' in line:
                    result.append(line.split('#', 1)[0])
                elif stripped.startswith('"""') or stripped.startswith("'''"):
                    in_multiline = True
                    if preserve_docstrings:
                        result.append(line)
                else:
                    result.append(line)

        return '\n'.join(result)

    def remove_comments(self, file_path: Path, source_code: str, preserve_docstrings: bool) -> str:
        """Универсальная точка входа для удаления комментариев (Python и др. языки)"""
        suffix = file_path.suffix.lower()
        if suffix == '.py':
            return self.remove_python_comments(source_code, preserve_docstrings)
        # TODO
        return source_code

    def detect_language(self, file_path: Path, code_content: str) -> str:
        """Определяем язык с помощью Pygments; при неудаче используем словарь расширений"""
        try:
            lexer = guess_lexer_for_filename(file_path.name, code_content)
            return lexer.aliases[0] if lexer.aliases else lexer.name.lower()
        except ClassNotFound:
            return self.extension_to_language.get(file_path.suffix.lower(), 'text')


class ProjectProcessor:
    def __init__(
        self,
        config: Dict,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.cancel_flag = threading.Event()
        self.code_processor = CodeProcessor()

        models = {**BUILTIN_MODEL_CONTEXT_SIZES, **self.config.get("custom_models", {})}
        selected_model = self.config.get("selected_model", "Gemini")
        self.max_tokens = models.get(selected_model, 2000000)

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.get("max_threads", 4)
        )

        self.chunk_size = self.config.get("chunk_size", 1024)

    def update_status(self, message: str) -> None:
        if self.status_callback:
            self.status_callback(message)
        logger.info(message)

    def update_progress(self, current: int, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(current, total)

    def cancel(self) -> None:
        self.cancel_flag.set()
        logger.info("Operation canceled by user.")

    def process_project(self, src_directory: Path, extensions: List[str]) -> Optional[Dict]:
        """Сканируем директорию, обрабатываем файлы в пуле потоков"""
        if self.cancel_flag.is_set():
            return None
        try:
            self.update_status(f"Processing project: {src_directory}")

            if not src_directory.exists() or not src_directory.is_dir():
                logger.error(f"Directory {src_directory} does not exist or is not a directory.")
                return None

            project_name = src_directory.name
            project_structure = []
            files_info = []
            total_files = 0
            processed_files = 0

            for root, directories, files in os.walk(src_directory):
                current_path = Path(root)
                try:
                    relative_parts = current_path.relative_to(src_directory).parts
                except ValueError:
                    continue

                if not self.config.get("include_libraries", False) and any(
                    part.startswith('.') or part == '__pycache__' for part in relative_parts
                ):
                    continue

                for file_name in files:
                    file_path = current_path / file_name
                    if file_path.suffix.lower() in extensions:
                        total_files += 1

            if total_files == 0:
                logger.warning(f"No matching files found in project {project_name}.")
                return {
                    'project_name': project_name,
                    'project_structure': [],
                    'files_info': [],
                    'project_path': src_directory
                }

            for root, directories, files in os.walk(src_directory):
                if self.cancel_flag.is_set():
                    return None

                current_path = Path(root)
                try:
                    relative_parts = current_path.relative_to(src_directory).parts
                except ValueError:
                    continue

                if not self.config.get("include_libraries", False) and any(
                    part.startswith('.') or part == '__pycache__' for part in relative_parts
                ):
                    continue

                level = len(relative_parts)
                indent = '  ' * level
                project_structure.append(f"{indent}- {current_path.name}/\n")

                future_to_file = {}
                for file_name in files:
                    file_path = current_path / file_name
                    if file_path.suffix.lower() in extensions:
                        if not self.code_processor.is_binary(file_path):
                            future = self.executor.submit(
                                self._process_file, file_path, src_directory
                            )
                            future_to_file[future] = file_path

                for future in concurrent.futures.as_completed(future_to_file):
                    if self.cancel_flag.is_set():
                        return None

                    file_path = future_to_file[future]
                    processed_files += 1

                    try:
                        file_info = future.result()
                        if file_info:
                            files_info.append(file_info)
                    except Exception as exc:
                        logger.error(f"Processing file {file_path} raised exception: {exc}", exc_info=True)

                    self.update_progress(processed_files, total_files)

            return {
                'project_name': project_name,
                'project_structure': project_structure,
                'files_info': files_info,
                'project_path': src_directory
            }

        except Exception as error:
            logger.error(f"Error processing project {src_directory}: {error}", exc_info=True)
            return None

    def _process_file(self, file_path: Path, project_path: Path) -> Optional[Dict]:
        """Обработка отдельного файла: чтение, удаление комментариев (если нужно), определение языка"""
        try:
            encoding = self.code_processor.detect_encoding(
                file_path,
                self.config.get("max_file_size_mb", 50)
            )
            if not encoding:
                return None

            code_content = self.code_processor.read_file_safely(file_path, encoding)
            if not code_content:
                return None

            if self.config.get("remove_comments", True):
                code_content = self.code_processor.remove_comments(
                    file_path,
                    code_content,
                    self.config.get("preserve_docstrings", True)
                )

            language = self.code_processor.detect_language(file_path, code_content)

            return {
                'file_path': file_path,
                'language': language,
                'code_content': code_content
            }
        except Exception as error:
            logger.error(f"Error processing file {file_path}: {error}", exc_info=True)
            return None

    def generate_reports(self, projects_info: List[Dict]) -> bool:
        """Создаём результирующие Markdown-отчёты (либо единый, либо несколько частей)"""
        if not projects_info:
            self.update_status("No projects to process.")
            return False

        try:
            self.update_status("Generating reports...")
            report_parts = []
            current_part = []
            current_tokens = 0

            initial_header = "# Unified Project Report\n\n"
            current_part.append(initial_header)
            current_tokens += count_tokens(initial_header)

            for project_index, project_info in enumerate(projects_info):
                if self.cancel_flag.is_set():
                    return False

                project_header = f"## Project: {project_info['project_name']}\n\n"
                project_structure = f"### Project Structure:\n{''.join(project_info['project_structure'])}\n\n"

                project_header_tokens = count_tokens(project_header)
                project_structure_tokens = count_tokens(project_structure)

                if current_tokens + project_header_tokens > self.max_tokens:
                    report_parts.append("".join(current_part))
                    current_part = [project_header]
                    current_tokens = project_header_tokens
                else:
                    current_part.append(project_header)
                    current_tokens += project_header_tokens

                if current_tokens + project_structure_tokens > self.max_tokens:
                    report_parts.append("".join(current_part))
                    current_part = [project_structure]
                    current_tokens = project_structure_tokens
                else:
                    current_part.append(project_structure)
                    current_tokens += project_structure_tokens

                total_files = len(project_info['files_info'])
                for file_index, file_info in enumerate(project_info['files_info']):
                    if self.cancel_flag.is_set():
                        return False

                    self.update_status(
                        f"Processing file {file_index + 1}/{total_files} in project {project_info['project_name']}"
                    )

                    try:
                        relative_file_path = file_info['file_path'].relative_to(project_info['project_path'])
                    except ValueError:
                        relative_file_path = file_info['file_path']

                    file_section = (
                        f"#### {relative_file_path}\n\n"
                        f"```{file_info['language']}\n{file_info['code_content']}\n```\n\n"
                    )

                    file_tokens = count_tokens_in_chunks(file_section, self.chunk_size)

                    if file_tokens > self.max_tokens:
                        code_lines = file_info['code_content'].splitlines(keepends=True)
                        chunk = []
                        chunk_tokens = 0
                        line_number = 1

                        for index, line in enumerate(code_lines, start=1):
                            line_tokens = count_tokens(line)

                            if chunk_tokens + line_tokens > self.max_tokens:
                                chunk_content = ''.join(chunk)
                                chunk_section = (
                                    f"#### {relative_file_path} (lines {line_number}-{index - 1})\n\n"
                                    f"```{file_info['language']}\n{chunk_content}\n```\n\n"
                                )
                                chunk_section_tokens = count_tokens_in_chunks(chunk_section, self.chunk_size)

                                if current_tokens + chunk_section_tokens > self.max_tokens:
                                    report_parts.append("".join(current_part))
                                    current_part = [chunk_section]
                                    current_tokens = chunk_section_tokens
                                else:
                                    current_part.append(chunk_section)
                                    current_tokens += chunk_section_tokens

                                chunk = [line]
                                chunk_tokens = line_tokens
                                line_number = index
                            else:
                                chunk.append(line)
                                chunk_tokens += line_tokens

                        if chunk:
                            chunk_content = ''.join(chunk)
                            chunk_section = (
                                f"#### {relative_file_path} (lines {line_number}-{index})\n\n"
                                f"```{file_info['language']}\n{chunk_content}\n```\n\n"
                            )
                            chunk_section_tokens = count_tokens_in_chunks(chunk_section, self.chunk_size)

                            if current_tokens + chunk_section_tokens > self.max_tokens:
                                report_parts.append("".join(current_part))
                                current_part = [chunk_section]
                                current_tokens = chunk_section_tokens
                            else:
                                current_part.append(chunk_section)
                                current_tokens += chunk_section_tokens
                    else:
                        if current_tokens + file_tokens > self.max_tokens:
                            report_parts.append("".join(current_part))
                            current_part = [file_section]
                            current_tokens = file_tokens
                        else:
                            current_part.append(file_section)
                            current_tokens += file_tokens

                    self.update_progress(
                        project_index * 100 + (file_index + 1) * 100 // max(total_files, 1),
                        len(projects_info) * 100
                    )

            if current_part:
                report_parts.append("".join(current_part))

            self._save_reports(report_parts, initial_header)
            return True

        except Exception as error:
            logger.error(f"Error generating reports: {error}", exc_info=True)
            return False

    def _save_reports(self, report_parts: List[str], initial_header: str) -> None:
        """Сохранение собранных частей отчетов на диск"""
        output_file_template = Path(self.config.get("output_file_template", "{project_name}_report.md"))
        unified_output_file = Path(self.config.get("unified_output_file", "unified_report.md"))
        unified_report = self.config.get("unified_report", True)

        self.update_status(f"Saving reports ({len(report_parts)} parts)...")

        for index, part in enumerate(report_parts):
            if self.cancel_flag.is_set():
                return

            try:
                if unified_report:
                    if len(report_parts) == 1:
                        part_file_path = unified_output_file
                    else:
                        part_file_path = unified_output_file.with_name(
                            f"{unified_output_file.stem}_part{index + 1}{unified_output_file.suffix}"
                        )
                    part_content = part
                else:
                    part_file_path = output_file_template.with_name(
                        f"{output_file_template.stem}_part{index + 1}{output_file_template.suffix}"
                    )
                    part_content = part

                part_file_path.parent.mkdir(parents=True, exist_ok=True)

                with part_file_path.open("w", encoding="utf-8") as file:
                    file.write(part_content)

                logger.info(f"Report part {index + 1} saved to {part_file_path}")

            except Exception as error:
                logger.error(f"Error writing to file {part_file_path}: {error}", exc_info=True)

        self.update_status("Reports saved. Processing complete.")

    def run(self) -> bool:
        """Главная точка запуска обработки"""
        try:
            self.cancel_flag.clear()

            source_directories = [
                Path(item["path"]) for item in self.config.get("src_directories", [])
                if item.get("selected", True) and Path(item["path"]).is_dir()
            ]
            if not source_directories:
                self.update_status("Source directory list is empty or inaccessible.")
                return False

            extensions = [ext.strip().lower() for ext in self.config.get("extensions", [])]
            if not extensions:
                self.update_status("File extension list is empty.")
                return False

            self.update_status(f"Starting to process {len(source_directories)} directories...")
            projects_info = []

            for index, source_directory in enumerate(source_directories):
                if self.cancel_flag.is_set():
                    self.update_status("Operation canceled by user.")
                    return False

                self.update_status(f"Processing project {index + 1}/{len(source_directories)}: {source_directory}")
                project_info = self.process_project(source_directory, extensions)
                if project_info:
                    projects_info.append(project_info)

                self.update_progress(index + 1, len(source_directories))

            if not self.cancel_flag.is_set():
                return self.generate_reports(projects_info)
            else:
                self.update_status("Operation canceled by user.")
                return False

        except Exception as error:
            logger.error(f"Error during processing: {error}", exc_info=True)
            self.update_status(f"An error occurred: {error}")
            return False
        finally:
            self.executor.shutdown(wait=False)


class DirectorySelectionFrame(ttk.Frame):
    """
    Фрейм для работы со списком директорий (добавление, удаление).
    """
    def __init__(self, parent, config_manager: ConfigManager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config_manager = config_manager
        self.vars: Dict[str, tk.BooleanVar] = {}
        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, height=200)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        self.inner_frame = ttk.Frame(self.canvas)
        self.inner_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.populate()

        self.bind("<Enter>", self._bind_mousewheel)
        self.bind("<Leave>", self._unbind_mousewheel)

    def _bind_mousewheel(self, event):
        if sys.platform.startswith('win'):
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        elif sys.platform.startswith('darwin'):
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_macos)
        else:
            self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, event):
        if sys.platform.startswith('win'):
            self.canvas.unbind_all("<MouseWheel>")
        elif sys.platform.startswith('darwin'):
            self.canvas.unbind_all("<MouseWheel>")
        else:
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_macos(self, event):
        self.canvas.yview_scroll(int(-1 * event.delta), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def populate(self):
        for child in self.inner_frame.winfo_children():
            child.destroy()
        self.vars.clear()

        for item in self.config_manager.config.get("src_directories", []):
            variable = tk.BooleanVar(value=item.get("selected", True))

            def on_variable_change(name, index, mode, item=item, var=variable):
                self.on_toggle(item, var)

            variable.trace_add("write", on_variable_change)

            frame = ttk.Frame(self.inner_frame)
            frame.pack(fill="x", padx=2, pady=1)

            checkbox = ttk.Checkbutton(
                frame,
                variable=variable,
                padding=0
            )
            checkbox.pack(side="left", padx=0, pady=0)

            path_label = ttk.Label(
                frame,
                text=item["path"],
                wraplength=400
            )
            path_label.pack(side="left", fill="x", expand=True, padx=2, pady=0, anchor="w")

            self.vars[item["path"]] = variable

    def on_toggle(self, item, variable: tk.BooleanVar):
        item["selected"] = variable.get()

    def add_directory(self, directory: str):
        if not directory:
            return

        if any(item["path"] == directory for item in self.config_manager.config.get("src_directories", [])):
            return

        self.config_manager.config.setdefault("src_directories", []).append(
            {"path": directory, "selected": True}
        )

        self.populate()

    def remove_selected(self):
        old_list = self.config_manager.config.get("src_directories", [])
        new_list = []
        for item in old_list:
            path = item["path"]
            var = self.vars.get(path)
            if var is not None and var.get():
                continue
            new_list.append(item)

        self.config_manager.config["src_directories"] = new_list
        self.populate()


class EditModelDialog(tk.Toplevel):
    """
    Диалог для добавления/редактирования одной модели.
    """
    def __init__(self, parent, title="Add Model", model_name="", token_limit=""):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None

        window_width = 300
        window_height = 150
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.create_widgets(model_name, token_limit)

        self.entry_name.focus_set()
        self.grab_set()

    def create_widgets(self, model_name, token_limit):
        label_name = ttk.Label(self, text="Model Name:")
        label_name.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.entry_name = ttk.Entry(self)
        self.entry_name.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        self.entry_name.insert(0, model_name)

        label_limit = ttk.Label(self, text="Token Limit:")
        label_limit.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.entry_limit = ttk.Entry(self)
        self.entry_limit.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        self.entry_limit.insert(0, token_limit)

        frame_buttons = ttk.Frame(self)
        frame_buttons.grid(row=2, column=0, columnspan=2, pady=5)

        button_ok = ttk.Button(frame_buttons, text="OK", command=self.on_ok)
        button_ok.pack(side="left", padx=5)

        button_cancel = ttk.Button(frame_buttons, text="Cancel", command=self.on_cancel)
        button_cancel.pack(side="left", padx=5)

        self.columnconfigure(1, weight=1)

        self.bind("<Return>", lambda event: self.on_ok())
        self.bind("<Escape>", lambda event: self.on_cancel())

    def on_ok(self):
        name = self.entry_name.get().strip()

        try:
            limit = int(self.entry_limit.get().strip())
            if limit <= 0:
                raise ValueError("Limit must be a positive number")
        except ValueError:
            messagebox.showerror("Error", "Token limit must be a positive integer.")
            return

        if not name:
            messagebox.showerror("Error", "Model name cannot be empty.")
            return

        self.result = (name, limit)
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()


class CustomModelsDialog(tk.Toplevel):
    """
    Окно для управления кастомными моделями (добавление, удаление, редактирование).
    """
    def __init__(self, parent, config_manager, update_callback=None):
        super().__init__(parent)
        self.title("Manage Custom Models")
        self.config_manager = config_manager
        self.update_callback = update_callback

        window_width = 500
        window_height = 400
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.create_widgets()
        self.refresh_treeview()
        self.grab_set()

    def create_widgets(self):
        self.tree = ttk.Treeview(
            self,
            columns=("name", "limit"),
            show="headings",
            selectmode="browse"
        )

        self.tree.heading("name", text="Model Name")
        self.tree.heading("limit", text="Token Limit")
        self.tree.column("name", width=200)
        self.tree.column("limit", width=100, anchor="center")

        self.tree.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=3, pady=5, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        button_add = ttk.Button(self, text="Add", command=self.add_model)
        button_add.grid(row=1, column=0, padx=5, pady=5, sticky="we")

        button_edit = ttk.Button(self, text="Edit", command=self.edit_model)
        button_edit.grid(row=1, column=1, padx=5, pady=5, sticky="we")

        button_delete = ttk.Button(self, text="Delete", command=self.delete_model)
        button_delete.grid(row=1, column=2, padx=5, pady=5, sticky="we")

        button_close = ttk.Button(self, text="Close", command=self.on_close)
        button_close.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="we")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.tree.bind("<Double-1>", lambda event: self.edit_model())

    def refresh_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        custom_models = self.config_manager.config.get("custom_models", {})
        for name, limit in custom_models.items():
            self.tree.insert("", "end", values=(name, limit))

        if self.tree.get_children():
            self.tree.selection_set(self.tree.get_children()[0])

    def add_model(self):
        dialog = EditModelDialog(self, title="Add Model")
        self.wait_window(dialog)

        if dialog.result:
            name, limit = dialog.result
            custom_models = self.config_manager.config.get("custom_models", {})
            if name in custom_models:
                messagebox.showerror("Error", "A model with this name already exists.")
            else:
                custom_models[name] = limit
                self.config_manager.config["custom_models"] = custom_models
                self.refresh_treeview()

    def edit_model(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a model to edit.")
            return

        item = self.tree.item(selected[0])
        current_name, current_limit = item["values"]

        dialog = EditModelDialog(
            self,
            title="Edit Model",
            model_name=current_name,
            token_limit=str(current_limit)
        )
        self.wait_window(dialog)

        if dialog.result:
            new_name, new_limit = dialog.result
            custom_models = self.config_manager.config.get("custom_models", {})

            if new_name != current_name and new_name in custom_models:
                messagebox.showerror("Error", "A model with this name already exists.")
                return

            del custom_models[current_name]
            custom_models[new_name] = new_limit
            self.config_manager.config["custom_models"] = custom_models
            self.refresh_treeview()

    def delete_model(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a model to delete.")
            return

        item = self.tree.item(selected[0])
        name = item["values"][0]

        if messagebox.askyesno("Confirmation", f"Delete model '{name}'?"):
            custom_models = self.config_manager.config.get("custom_models", {})
            if name in custom_models:
                del custom_models[name]
                self.config_manager.config["custom_models"] = custom_models
                self.refresh_treeview()

    def on_close(self):
        self.config_manager.save_config()
        if self.update_callback:
            self.update_callback()
        self.destroy()


class App(TKMT.ThemedTKinterFrame):
    """
    Основное приложение с использованием TkinterModernThemes.
    Слева панель с настройками (Tab-ы), справа — лог и прогресс.
    """
    def __init__(self, config_manager, theme="sun-valley", mode="dark"):
        super().__init__("Code Report Generator", theme, mode)
        self.config_manager = config_manager
        self.processor = None

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(950, screen_width - 100)
        window_height = min(700, screen_height - 100)
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.create_widgets()
        self.poll_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.minsize(800, 600)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.paned_window = ttk.PanedWindow(main_frame, orient="horizontal")
        self.paned_window.pack(fill='both', expand=True)

        self.settings_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.settings_frame, weight=1)

        self.log_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.log_frame, weight=1)

        self._create_settings_section()
        self._create_logs_section()

    def _create_settings_section(self):
        title_label = ttk.Label(self.settings_frame, text="Project Settings", font=("", 12, "bold"))
        title_label.pack(fill="x", padx=5, pady=(5, 10))

        settings_notebook = ttk.Notebook(self.settings_frame)
        settings_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        directories_frame = ttk.Frame(settings_notebook)
        files_frame = ttk.Frame(settings_notebook)
        output_frame = ttk.Frame(settings_notebook)
        models_frame = ttk.Frame(settings_notebook)

        settings_notebook.add(directories_frame, text="Directories")
        settings_notebook.add(files_frame, text="Files")
        settings_notebook.add(output_frame, text="Output")
        settings_notebook.add(models_frame, text="Models")

        self._create_directories_tab(directories_frame)
        self._create_files_tab(files_frame)
        self._create_output_tab(output_frame)
        self._create_models_tab(models_frame)

        button_frame = ttk.Frame(self.settings_frame)
        button_frame.pack(fill="x", padx=5, pady=10)

        self.button_start = ttk.Button(
            button_frame,
            text="Start Processing",
            command=self.start_processing
        )
        self.button_start.pack(side="left", expand=True, fill="x", padx=5)

        self.button_cancel = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_processing,
            state="disabled"
        )
        self.button_cancel.pack(side="left", expand=True, fill="x", padx=5)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.settings_frame, textvariable=self.status_var, font=("", 9, "italic"))
        status_label.pack(fill="x", padx=5, pady=5)

    def _create_directories_tab(self, parent):
        frame_directories = ttk.LabelFrame(parent, text="Source Directories")
        frame_directories.pack(fill="both", expand=True, padx=5, pady=5)

        self.dir_frame = DirectorySelectionFrame(frame_directories, self.config_manager)
        self.dir_frame.pack(fill="both", expand=True, padx=5, pady=5)

        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", padx=5, pady=5)

        button_add_directory = ttk.Button(
            button_frame,
            text="Add Directory",
            command=self.add_directory
        )
        button_add_directory.pack(side="left", expand=True, fill="x", padx=5)

        button_remove_directory = ttk.Button(
            button_frame,
            text="Remove Selected",
            command=self.remove_directory
        )
        button_remove_directory.pack(side="left", expand=True, fill="x", padx=5)

        self.include_libraries_var = tk.BooleanVar(value=self.config_manager.config.get("include_libraries", False))
        checkbox_include_libraries = ttk.Checkbutton(
            parent,
            text="Include Library Directories",
            variable=self.include_libraries_var
        )
        checkbox_include_libraries.pack(anchor="w", padx=10, pady=5)

    def _create_files_tab(self, parent):
        frame_extensions = ttk.LabelFrame(parent, text="File Extensions (comma separated)")
        frame_extensions.pack(fill="x", padx=5, pady=5)

        self.entry_extensions = ttk.Entry(frame_extensions)
        self.entry_extensions.pack(fill="x", padx=5, pady=5)
        self.entry_extensions.insert(
            0,
            ", ".join(self.config_manager.config.get("extensions", []))
        )

        frame_comments = ttk.LabelFrame(parent, text="Comment Removal Options")
        frame_comments.pack(fill="x", padx=5, pady=5)

        self.remove_comments_var = tk.BooleanVar(value=self.config_manager.config.get("remove_comments", True))
        checkbox_remove_comments = ttk.Checkbutton(
            frame_comments,
            text="Remove Comments from Code",
            variable=self.remove_comments_var
        )
        checkbox_remove_comments.pack(anchor="w", padx=5, pady=5)

        self.preserve_docstrings_var = tk.BooleanVar(value=self.config_manager.config.get("preserve_docstrings", True))
        checkbox_preserve_docstrings = ttk.Checkbutton(
            frame_comments,
            text="Preserve Docstrings",
            variable=self.preserve_docstrings_var
        )
        checkbox_preserve_docstrings.pack(anchor="w", padx=5, pady=5)

        frame_size = ttk.LabelFrame(parent, text="File Size Limit (MB)")
        frame_size.pack(fill="x", padx=5, pady=5)

        self.spinner_max_file_size = ttk.Spinbox(
            frame_size,
            from_=1,
            to=100,
            increment=1
        )
        self.spinner_max_file_size.pack(fill="x", padx=5, pady=5)
        self.spinner_max_file_size.set(self.config_manager.config.get("max_file_size_mb", 50))

    def _create_output_tab(self, parent):
        frame_output = ttk.LabelFrame(parent, text="Output File Templates")
        frame_output.pack(fill="x", padx=5, pady=5)

        label_output_template = ttk.Label(frame_output, text="Per-project output file:")
        label_output_template.pack(anchor="w", padx=5, pady=2)

        self.entry_output_template = ttk.Entry(frame_output)
        self.entry_output_template.pack(fill="x", padx=5, pady=2)
        self.entry_output_template.insert(
            0,
            self.config_manager.config.get("output_file_template", "{project_name}_report.md")
        )

        label_unified_output = ttk.Label(frame_output, text="Unified report file:")
        label_unified_output.pack(anchor="w", padx=5, pady=2)

        self.entry_unified_output = ttk.Entry(frame_output)
        self.entry_unified_output.pack(fill="x", padx=5, pady=2)
        self.entry_unified_output.insert(
            0,
            self.config_manager.config.get("unified_output_file", "unified_report.md")
        )

        frame_report_options = ttk.LabelFrame(parent, text="Report Options")
        frame_report_options.pack(fill="x", padx=5, pady=5)

        self.unified_report_var = tk.BooleanVar(value=self.config_manager.config.get("unified_report", True))
        checkbox_unified_report = ttk.Checkbutton(
            frame_report_options,
            text="Generate Unified Report",
            variable=self.unified_report_var
        )
        checkbox_unified_report.pack(anchor="w", padx=5, pady=5)

        frame_performance = ttk.LabelFrame(parent, text="Performance Settings")
        frame_performance.pack(fill="x", padx=5, pady=5)

        label_max_threads = ttk.Label(frame_performance, text="Maximum threads:")
        label_max_threads.pack(anchor="w", padx=5, pady=2)

        self.spinner_max_threads = ttk.Spinbox(
            frame_performance,
            from_=1,
            to=16,
            increment=1,
            width=5
        )
        self.spinner_max_threads.pack(anchor="w", padx=5, pady=2)
        self.spinner_max_threads.set(self.config_manager.config.get("max_threads", 4))

    def _create_models_tab(self, parent):
        frame_model = ttk.LabelFrame(parent, text="Model Selection")
        frame_model.pack(fill="x", padx=5, pady=5)

        label_model = ttk.Label(frame_model, text="Select model:")
        label_model.pack(anchor="w", padx=5, pady=2)

        self.combo_model = ttk.Combobox(frame_model)
        self.combo_model.pack(fill="x", padx=5, pady=2)
        self.update_model_combobox()

        button_manage_models = ttk.Button(
            frame_model,
            text="Manage Custom Models",
            command=self.manage_models
        )
        button_manage_models.pack(fill="x", padx=5, pady=5)

        frame_logging = ttk.LabelFrame(parent, text="Logging")
        frame_logging.pack(fill="x", padx=5, pady=5)

        label_log_level = ttk.Label(frame_logging, text="Log level:")
        label_log_level.pack(anchor="w", padx=5, pady=2)

        self.combo_log_level = ttk.Combobox(
            frame_logging,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        self.combo_log_level.pack(fill="x", padx=5, pady=2)
        self.combo_log_level.set(self.config_manager.config.get("log_level", "INFO"))

    def _create_logs_section(self):
        title_label = ttk.Label(self.log_frame, text="Logs & Progress", font=("", 12, "bold"))
        title_label.pack(fill="x", padx=5, pady=(5, 10))

        self.text_log = scrolledtext.ScrolledText(self.log_frame, wrap='none', state='disabled', height=20)
        self.text_log.pack(fill='both', expand=True, padx=5, pady=5)

        progress_frame = ttk.Frame(self.log_frame)
        progress_frame.pack(fill='x', padx=5, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Waiting", anchor="w")
        self.progress_label.pack(fill='x', padx=5, pady=2)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=5, pady=2)

        self.button_clear_log = ttk.Button(
            progress_frame,
            text="Clear Logs",
            command=self.clear_logs
        )
        self.button_clear_log.pack(anchor="e", padx=5, pady=5)

    def update_model_combobox(self):
        models = set(BUILTIN_MODEL_CONTEXT_SIZES.keys())
        custom = self.config_manager.config.get("custom_models", {})
        models.update(custom.keys())
        models_list = sorted(models)

        self.combo_model["values"] = models_list

        current = self.combo_model.get()
        if not current or current not in models_list:
            self.combo_model.set("Gemini")

    def manage_models(self):
        CustomModelsDialog(self.root, self.config_manager, update_callback=self.update_model_combobox)

    def add_directory(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.dir_frame.add_directory(directory_path)

    def remove_directory(self):
        self.dir_frame.remove_selected()

    def clear_logs(self):
        self.text_log.configure(state='normal')
        self.text_log.delete(1.0, tk.END)
        self.text_log.configure(state='disabled')

    def start_processing(self):
        source_directories = [
            item for item in self.config_manager.config.get("src_directories", [])
            if item.get("selected", True)
        ]

        if not source_directories:
            messagebox.showwarning("Warning", "No directories selected.")
            return

        extensions_text = self.entry_extensions.get().strip()
        if not extensions_text:
            messagebox.showwarning("Warning", "No file extensions specified.")
            return

        # Сохраняем настройки в конфиг
        self.config_manager.config["output_file_template"] = self.entry_output_template.get().strip()
        self.config_manager.config["unified_output_file"] = self.entry_unified_output.get().strip()

        extensions = [ext.strip() for ext in extensions_text.split(',') if ext.strip()]
        self.config_manager.config["extensions"] = extensions

        self.config_manager.config["include_libraries"] = self.include_libraries_var.get()
        self.config_manager.config["remove_comments"] = self.remove_comments_var.get()
        self.config_manager.config["preserve_docstrings"] = self.preserve_docstrings_var.get()
        self.config_manager.config["unified_report"] = self.unified_report_var.get()
        self.config_manager.config["log_level"] = self.combo_log_level.get()
        self.config_manager.config["selected_model"] = self.combo_model.get()

        try:
            max_threads = int(self.spinner_max_threads.get())
            if max_threads < 1:
                max_threads = 1
            elif max_threads > 16:
                max_threads = 16
            self.config_manager.config["max_threads"] = max_threads
        except ValueError:
            self.config_manager.config["max_threads"] = 4

        try:
            max_file_size = int(self.spinner_max_file_size.get())
            if max_file_size < 1:
                max_file_size = 1
            elif max_file_size > 100:
                max_file_size = 100
            self.config_manager.config["max_file_size_mb"] = max_file_size
        except ValueError:
            self.config_manager.config["max_file_size_mb"] = 50

        logger.setLevel(getattr(logging, self.config_manager.config["log_level"].upper(), logging.INFO))
        self.config_manager.save_config()
        self._set_processing_state(True)

        self.processor = ProjectProcessor(
            self.config_manager.config,
            progress_callback=self.update_progress,
            status_callback=self.update_status
        )

        threading.Thread(target=self._run_processing, daemon=True).start()

    def _run_processing(self):
        try:
            result = self.processor.run()
            self.root.after(100, lambda: self._processing_completed(result))
        except Exception as error:
            logger.error(f"Error during processing: {error}", exc_info=True)
            self.root.after(100, lambda: self._processing_completed(False, str(error)))

    def _processing_completed(self, success: bool, error_message: str = None):
        self._set_processing_state(False)

        if success:
            self.status_var.set("Processing completed successfully")
            self.progress_label.config(text="Completed")
        else:
            self.status_var.set(f"Error: {error_message}" if error_message else "Processing interrupted")
            self.progress_label.config(text="Interrupted")

            if error_message:
                messagebox.showerror("Error", f"An error occurred during processing: {error_message}")

        self.processor = None

    def cancel_processing(self):
        if self.processor:
            self.processor.cancel()
            self.status_var.set("Canceling operation...")
            self.progress_label.config(text="Canceling...")

    def _set_processing_state(self, is_processing: bool):
        if is_processing:
            self.button_start.configure(state="disabled")
            self.button_cancel.configure(state="normal")
            self.progress_var.set(0)
            self.status_var.set("Processing...")
            self.progress_label.config(text="Preparing...")
        else:
            self.button_start.configure(state="normal")
            self.button_cancel.configure(state="disabled")

    def update_progress(self, current: int, total: int):
        progress_percent = (current / total) * 100 if total else 0
        self.progress_var.set(progress_percent)
        self.progress_label.config(text=f"Progress: {current}/{total} ({progress_percent:.1f}%)")

    def update_status(self, message: str):
        self.status_var.set(message)
        self.progress_label.config(text=message)

    def poll_log_queue(self):
        while not log_queue.empty():
            try:
                message = log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.append_log(message)
        self.root.after(100, self.poll_log_queue)

    def append_log(self, message: str):
        self.text_log.configure(state='normal')
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.configure(state='disabled')
        self.text_log.see(tk.END)

    def on_close(self):
        if self.processor and not self.processor.cancel_flag.is_set():
            if messagebox.askyesno("Confirmation", "Processing is not complete. Are you sure you want to exit?"):
                if self.processor:
                    self.processor.cancel()
                self.config_manager.save_config()
                self.root.destroy()
        else:
            self.config_manager.save_config()
            self.root.destroy()


if __name__ == "__main__":
    theme = config_manager.config.get("ui_theme", "dark")
    app = App(config_manager, theme="sun-valley", mode=theme)
    app.run()
